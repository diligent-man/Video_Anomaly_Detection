from typing import Tuple

import torch

from .QKVProj import QKVProj


__all__ = ["RelPosProj"]


class RelPosProj(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 max_rel_pos: int,
                 bias: bool = True,
                 device: torch.device = None,
                 dtype: torch.dtype = None,
                 ) -> None:
        super(RelPosProj, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.__embed_dim: int = embed_dim
        self.__max_rel_pos: int = max_rel_pos
        self.__bias: bool = bias
        self.__device: torch.device = device

        self.__qkv_proj: QKVProj = QKVProj(self.__embed_dim, self.__bias, **factory_kwargs)

        # Shared rel pos embeddings
        self.__rel_pos_embed: torch.nn.Embedding = torch.nn.Embedding(2 * self.__max_rel_pos, embed_dim, **factory_kwargs)

    def forward(self, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        :param seq_len: length of sequence
        :return: Q, K w.r.t relative position embeddings. Shape [seq_len, seq_len, embed_dim]
        """
        assert self.__max_rel_pos <= seq_len, ValueError("Max rel pos exceeds sequence length.")

        # [seq_len, ]
        rel_pos: torch.Tensor = torch.arange(seq_len, dtype=torch.int32, device=self.__device)

        # [seq_len, seq_len]
        rel_pos = rel_pos.reshape((1, -1)) - rel_pos.reshape((-1, 1))

        # Rescale to [0, 2 * max_rel_pos - 1)
        rel_pos += self.__max_rel_pos
        rel_pos = rel_pos.clamp(0, 2 * self.__max_rel_pos - 1)

        rel_embeddings = self.__rel_pos_embed(rel_pos)
        rel_q, rel_k, _ = self.__qkv_proj(rel_embeddings)
        return rel_q, rel_k
