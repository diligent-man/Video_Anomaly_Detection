from typing import List, Dict

import torch


from .QKV import QKV
from .RelativePE import RelativePE
from .._functional import dynamic_expand


__all__ = ["TAM"]


class TAM(torch.nn.Module):
    def __init__(self,
                 num_backbones: int,
                 relative_attention: bool = True,
                 max_relative_position: int = 10,
                 embed_dim: int = 1024
                 ):
        super().__init__()

        assert max_relative_position >= 0, ValueError("max_relative_positions must be >= 0")
        assert embed_dim >= 0, ValueError("hidden_size must be > 0")

        self._embed_dim = embed_dim
        self._num_backbones = num_backbones
        self._content_qkv = torch.nn.ModuleList([QKV(embed_dim) for _ in range(num_backbones)])

        if relative_attention:
            if max_relative_position == 0:
                max_relative_position = embed_dim
            self._relative_pe = RelativePE(embed_dim, max_relative_position)
        else:
            self.register_parameter("_relative_pe", None)

    def _compute_hidden_state(self,
                              seq_len: int,
                              q: torch.Tensor,
                              k: torch.Tensor,
                              v: torch.Tensor,
                              next_k: torch.Tensor = None
                              ) -> torch.Tensor:
        """
        :param seq_len: sequence length
        :param q: query of current seq. Shape [batch_size, seq_len, embed_dim]
        :param k: key of current seq. Shape [batch_size, seq_len, embed_dim]
        :param v: value of current seq. Shape [batch_size, seq_len, embed_dim]
        :param next_k: key of next seq for calculating cross-attn. Shape [batch_size, seq_len, embed_dim]
        :return: hidden state of current sequence. Shape [batch_size, seq_len, embed_dim]

        Einsum ref: https://stackoverflow.com/questions/55894693/understanding-pytorch-einsum
        Algo ref: https://towardsdatascience.com/large-language-models-deberta-decoding-enhanced-bert-with-disentangled-attention-90016668db4b
        Note: Multi-head is currently not implemented
        """

        if self._relative_pe is not None:
            # rel_q, rel_k: [attn_span, embed_dim],
            # rel_pos_idx: [seq_len, seq_len]
            rel_q, rel_k, c2p_rel_pos, p2c_rel_pos = self._relative_pe(seq_len)

            # [batch, seq_len, hidden_dim] x [1, attn_span, embed_dim]
            c2p_attn: torch.Tensor = q @ rel_k.transpose(-1,-2)
            c2p_attn = torch.gather(c2p_attn, dim=-1, index=dynamic_expand(c2p_rel_pos, q, [0, 1, 1]))

            # [batch, seq_len, hidden_dim] x [1, attn_span, embed_dim]
            p2c_attn: torch.Tensor = k @ rel_q.transpose(-1, -2)
            p2c_attn = torch.gather(p2c_attn, dim=-1, index=dynamic_expand(p2c_rel_pos, k, [0, 1, 1]))
        else:
            c2p_attn, p2c_attn = torch.tensor(0), torch.tensor(0)

        # 4 attn scores return shape [batch, seq_len, seq_len]
        c2c_attn: torch.Tensor = q @ k.transpose(-2, -1)  # self-attn
        cross_c2c_attn: torch.Tensor | None = q @ next_k.transpose(-2, -1) if next_k is not None else torch.tensor(0) # cross-attn

        attn_score: torch.Tensor = c2c_attn + cross_c2c_attn + c2p_attn + p2c_attn

        attn_score /= (((self._num_backbones + 2) * self._embed_dim) ** .5)
        attn_score = attn_score.softmax(dim=-1)
        return attn_score @ v

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: hidden states of shape (num_backbones, batch_size, seq_len, embed_dim)
        :return: embedded tensor of shape [batch_size, seq_len, embed_dim]
        """
        assert x.dim() == 4, "Required dimension is not satisfied"
        num_backbones, _, seq_len, embed_dim = x.size()

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
                    hidden_state: torch.Tensor = self._compute_hidden_state(seq_len, *cache["last_backbone"], cache["first_k"])
                else:
                    hidden_state: torch.Tensor = self._compute_hidden_state(seq_len, *cache["last_backbone"], current_backbone[1])

                if output is None:
                    output = hidden_state
                else:
                    output += hidden_state

                cache["last_backbone"] = current_backbone

            output /= self._num_backbones
        return output
