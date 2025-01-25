from typing import List, Dict

import torch

from AI.src.model.CTA.Proj import QKVProj, RelPosProj


__all__ = ["CTAEncoder"]


class CTAEncoder(torch.nn.Module):
    def __init__(self,
                 num_backbones: int,
                 relative_attention: bool = True,
                 max_relative_position: int = 0,
                 embed_dim: int = 512
                 ):
        super().__init__()
        assert max_relative_position >= 0, ValueError("max_relative_positions must be >= 0")
        assert embed_dim >= 0, ValueError("hidden_size must be > 0")

        self._num_backbones = num_backbones
        self._content_proj_lst = torch.nn.ModuleList([QKVProj(embed_dim) for _ in range(num_backbones)])
        self._embed_dim = embed_dim

        if relative_attention:
            if max_relative_position == 0:
                max_relative_position = embed_dim
            self._pos_proj = RelPosProj(embed_dim, max_relative_position)
        else:
            self.register_parameter("__pos_proj", None)

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
        """
        # [seq_len, seq_len, embed_dim]
        rel_q, rel_k = self._pos_proj(seq_len)

        # TODO: Check how to calculate c2p, p2c ?
        # 4 attn scores return shape [batch, seq_len, seq_len]
        c2c_attn: torch.Tensor = q @ k.transpose(-2, -1)
        cross_c2c_attn: torch.Tensor | None = q @ next_k.transpose(-2, -1) if next_k is not None else None

        # [seq_len, seq_len, embed_dim] x [batch, seq_len, embed_dim]
        c2p_attn: torch.Tensor = torch.einsum("qkd, bkd -> bqk", rel_k, q)
        p2c_attn: torch.Tensor = torch.einsum("qkd, bkd -> bqk", rel_q, k)

        attn_score: torch.Tensor = c2c_attn + c2p_attn + p2c_attn
        if cross_c2c_attn is not None:
            attn_score += cross_c2c_attn

        attn_score /= (((self._num_backbones + 2) * self._embed_dim) ** .5)
        hidden_state: torch.Tensor = torch.einsum("bqk,bvd->bqd", attn_score.softmax(dim=-1), v)
        return hidden_state

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
            q, k, v = self._qkv_proj_lst[0](x)  # [batch_size, seq_len, embed_dim]
            output = self._compute_hidden_state(seq_len, q, k, v)
        else:
            hidden_state: None = None
            cache: Dict[str, torch.Tensor | List[torch.Tensor]] = {}

            for i in range(self._num_backbones):
                q, k, v = self._content_proj_lst[i](x[i])  # [batch_size, seq_len, embed_dim]
                current_backbone = [q, k, v]

                if i == 0:
                    cache["first_k"] = k
                elif i == self._num_backbones:
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
