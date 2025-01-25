from typing import Tuple

import torch

from .QKVProj import QKVProj


__all__ = ["RelPosProj"]


class RelPosProj(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 max_relative_positions: int,
                 bias: bool = True,
                 device: torch.device = None,
                 dtype: torch.dtype = None,
                 ) -> None:
        super(RelPosProj, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self._embed_dim: int = embed_dim
        self._max_relative_positions: int = max_relative_positions
        self._bias: bool = bias
        self._device: torch.device = device
        self._qkv_proj: QKVProj = QKVProj(self._embed_dim, self._bias, **factory_kwargs)

        # Shared rel pos embeddings
        self._rel_pos_embed: torch.nn.Embedding = torch.nn.Embedding(2 * self._max_relative_positions, embed_dim, **factory_kwargs)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param seq_len: length of sequence
        :return: Q, K w.r.t relative position embeddings. Shape [seq_len, seq_len, embed_dim]
        """
        assert self._max_relative_positions <= seq_len, ValueError("Max rel pos exceeds sequence length.")

        # [seq_len, ]
        rel_pos: torch.Tensor = torch.arange(seq_len, dtype=torch.int32, device=self._device)

        # [seq_len, seq_len]
        rel_pos = rel_pos.reshape((1, -1)) - rel_pos.reshape((-1, 1))

        # Rescale to [0, 2 * max_relative_positions - 1)
        rel_pos += 2 * self._max_relative_positions - 1
        rel_pos = rel_pos.clamp(0, 2 * self._max_relative_positions - 1).to(self._rel_pos_embed.weight.device)
        rel_embeddings = self._rel_pos_embed(rel_pos)
        rel_q, rel_k, _ = self._qkv_proj(rel_embeddings)
        return rel_q, rel_k
