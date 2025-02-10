from typing import List, Dict

import torch


from .QKV import QKV
from .RelativePE import RelativePE
from ..functional import dynamic_expand, transform_multihead


__all__ = ["TemporalAggregation"]


class TemporalAggregation(torch.nn.Module):
    def __init__(self,
                 num_backbones: int,
                 num_heads: int,
                 in_channels: int | List[int],
                 bias: bool = True,
                 relative_attention: bool = True,
                 max_relative_position: int = 10,
                 ):
        super().__init__()
        embed_dim: List[int] = [in_channels] if isinstance(in_channels, int) else in_channels

        assert max_relative_position >= 0, ValueError("max_relative_positions must be >= 0")
        for i in embed_dim:
            assert i >= 0, ValueError(f"Embed_dim must be > 0. Get '{i}'")
            assert i % num_heads == 0, ValueError(f"Embed_dim must be divisible by num_heads. Get '{i}' % {num_heads}")

        self._out_channels: int | List[int] = embed_dim
        self._num_backbones: int = num_backbones
        self._num_heads: int = num_heads
        self._embed_dim: List[int] = embed_dim
        self._content_qkv: torch.nn.ModuleList = torch.nn.ModuleList([QKV(x, bias) for x in self._embed_dim])

        if relative_attention:
            assert int(sum(self._embed_dim) // len(self._embed_dim)) == self._embed_dim[0], \
                ValueError("All embed_dim must equal when relative_attention is True")

            self._embed_dim: int = self._embed_dim[0]
            self._out_channels = self._embed_dim

            if max_relative_position == 0:
                max_relative_position = embed_dim

            self._relative_pe = RelativePE(self._embed_dim, max_relative_position, bias)
        else:
            self.register_parameter("_relative_pe", None)

    @property
    def out_channels(self) -> int | List[int]:
        return self._out_channels

    def _compute_hidden_state(self,
                              q: torch.Tensor,
                              k: torch.Tensor,
                              v: torch.Tensor,
                              next_k: torch.Tensor = None
                              ) -> torch.Tensor:
        """
        :param q: query of current seq. Shape [batch_size, seq_len, embed_dim]
        :param k: key of current seq. Shape [batch_size, seq_len, embed_dim]
        :param v: value of current seq. Shape [batch_size, seq_len, embed_dim]
        :param next_k: key of next seq for calculating cross-attn. Shape [batch_size, seq_len, embed_dim]
        :return: hidden state of current sequence. Shape [batch_size, seq_len, embed_dim]

        Einsum ref: https://stackoverflow.com/questions/55894693/understanding-pytorch-einsum
        Algo ref: https://towardsdatascience.com/large-language-models-deberta-decoding-enhanced-bert-with-disentangled-attention-90016668db4b
        Note: Multi-head is currently not implemented
        """
        batch, seq_len, embed_dim = v.shape
        q, k, v = [transform_multihead(x, self._num_heads) for x in [q, k, v]]  # [batch_size, num_heads, seq_len, head_dim]
        next_k = transform_multihead(next_k, self._num_heads) if next_k is not None else next_k

        if self._relative_pe is not None:
            # rel_q, rel_k: [attn_span, embed_dim],
            # rel_pos_idx: [seq_len, seq_len]
            rel_q, rel_k, c2p_rel_pos, p2c_rel_pos = self._relative_pe(seq_len)
            rel_q, rel_k = transform_multihead(rel_q, self._num_heads), transform_multihead(rel_k, self._num_heads)

            # [batch_size, num_heads, seq_len, head_dim] x [batch_size, num_heads, seq_len, head_dim]
            c2p_attn: torch.Tensor = q @ rel_k.transpose(-1, -2)
            c2p_attn = torch.gather(c2p_attn, dim=-1, index=dynamic_expand(c2p_rel_pos, q, [0, 1, 2, 2]))

            # [batch, seq_len, hidden_dim] x [1, attn_span, embed_dim]
            p2c_attn: torch.Tensor = k @ rel_q.transpose(-1, -2)
            p2c_attn = torch.gather(p2c_attn, dim=-1, index=dynamic_expand(p2c_rel_pos, k, [0, 1, 2, 2]))
        else:
            c2p_attn, p2c_attn = torch.tensor(0), torch.tensor(0)

        # 4 attn scores return shape [batch, seq_len, seq_len]
        c2c_attn: torch.Tensor = q @ k.transpose(-2, -1)  # self-attn
        cross_c2c_attn: torch.Tensor | None = q @ next_k.transpose(-2, -1) if next_k is not None else torch.tensor(0)  # cross-attn

        attn_score: torch.Tensor = c2c_attn + cross_c2c_attn + c2p_attn + p2c_attn
        attn_score /= (((self._num_backbones + 2) * self._embed_dim) ** .5)
        attn_score = attn_score.softmax(dim=-1)

        hidden_state = attn_score @ v
        # [batch, num_heads, seq_len, head_dim] -> [batch, seq_len, num_heads, head_dim] -> [batch, seq_len, embed_dim]
        hidden_state = hidden_state.permute(0, 2, 1, 3).reshape(batch, seq_len, embed_dim)
        return hidden_state

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: hidden states of shape (num_backbones, batch_size, seq_len, embed_dim)
        :return: embedded tensor of shape [batch_size, seq_len, embed_dim]
        """
        num_backbones, _, seq_len, embed_dim = x.size()
        assert x.dim() == 4, "Required dimension is not satisfied"
        assert self._num_backbones == num_backbones, "Input tensor has different number of backbones"

        output: torch.Tensor | None = None
        if self._num_backbones == 1:
            x = x.squeeze(dim=0)
            q, k, v = self._content_qkv[0](x)  # [batch_size, seq_len, hidden_dim]
            output = self._compute_hidden_state(seq_len, q, k, v)
        else:
            hidden_state: None = None
            cache: Dict[str, torch.Tensor | List[torch.Tensor]] = {}

            for i in range(self._num_backbones):
                q, k, v = self._content_qkv[i](x[i])  # [batch_size, seq_len, hidden_dim]
                current_backbone = [q, k, v]

                if i == 0:
                    cache["first_k"] = k
                elif i == self._num_backbones - 1:
                    hidden_state: torch.Tensor = self._compute_hidden_state(*cache["last_backbone"], cache["first_k"])
                else:
                    hidden_state: torch.Tensor = self._compute_hidden_state(*cache["last_backbone"], current_backbone[1])

                output = hidden_state if output is None else output + hidden_state
                cache["last_backbone"] = current_backbone
            output /= self._num_backbones
        return output
