from typing import Tuple

import torch
from multimethod import multimethod

from .QKV import QKV


__all__ = ["RelativePE"]


class RelativePE(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 max_rel_pos: int,
                 bias: bool = True,
                 device: torch.device = None,
                 dtype: torch.dtype = None,
                 ) -> None:
        super(RelativePE, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self._embed_dim: int = embed_dim
        self._max_rel_pos: int = max_rel_pos
        self._bias: bool = bias
        self._device: torch.device = device
        self._qkv: QKV = QKV(self._embed_dim, self._bias, **factory_kwargs)

        # Shared rel pos embeddings
        self._rel_pos_embed: torch.nn.Embedding = torch.nn.Embedding(2 * self._max_rel_pos, embed_dim, **factory_kwargs)

    @multimethod
    def _compute_attn_span(self, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
        """
        :param query: [batch, seq_len, q_hidden_dim]
        :param key: [batch, seq_len, k_hidden_dim]
        :return:
        """
        return torch.tensor(min(max(query.size(-2), key.size(-2)), self._max_rel_pos))

    @multimethod
    def _compute_attn_span(self, seq_len: int) -> torch.Tensor:
        return torch.tensor(min(seq_len, self._max_rel_pos))

    @multimethod
    def _get_rel_pos_idx(self, seq_len: int, attn_span: torch.Tensor, rel_type: str) -> torch.Tensor:
        """
        :param seq_len: sequence length
        :param attn_span: the span of relative position of query w.r.t key
        :param rel_type: "c2p" | "c2p".
        :return: relative indices b/t query and key

        Build relative position according to sequence length

        We assume the absolute position of query Pq is range from (0, query_size) and the absolute position of key
        Pk is range from (0, key_size), The relative positions from query to key is Rp->k = Pq - Pk.
        This method assumes for self-attn case
        """
        device = self._rel_pos_embed.weight.device

        rel_pos: torch.Tensor = torch.arange(seq_len, dtype=torch.long, device=device)  # [seq_len, ]
        rel_pos = rel_pos.reshape((-1, 1)) - rel_pos.reshape((1, -1))  # [seq_len, seq_len]

        if rel_type == "c2p":
            rel_pos = rel_pos + attn_span
        elif rel_type == "p2c":
            rel_pos = -rel_pos + attn_span

        rel_pos = torch.clamp(rel_pos, 0, 2 * attn_span.item() - 1)
        return rel_pos

    @multimethod
    def _get_rel_pos_idx(self, query: torch.Tensor, key: torch.Tensor, attn_span: torch.Tensor, rel_type: str) -> torch.Tensor:
        """
        :param query: embedded query of shape [Batch, Seq_len, Embed_dim]
        :param key: embedded key of shape [Batch, Seq_len, Embed_dim]
        :param attn_span: the span of relative position of query w.r.t key
        :return relative indices b/t query and key
        
        Build relative position according to the query and key

        We assume the absolute position of query Pq is range from (0, query_size) and the absolute position of key
        Pk is range from (0, key_size), The relative positions from query to key is Rp->k = Pq - Pk
        """
        query_dim, key_dim = query.size(-2), key.size(-2)  # Different in cross-attn circumstance
        
        q_ids: torch.Tensor = torch.arange(query_dim, dtype=torch.long)
        k_ids: torch.Tensor = torch.arange(key_dim, dtype=torch.long)

        rel_pos: torch.Tensor = q_ids.reshape((-1, 1)) - k_ids.reshape((1, -1))
        rel_pos = rel_pos.reshape((-1, 1)) - rel_pos.reshape((1, -1))  # [seq_len, seq_len]

        if rel_type == "c2p":
            rel_pos = rel_pos + attn_span
        elif rel_type == "p2c":
            rel_pos = -rel_pos + attn_span

        rel_pos = torch.clamp(rel_pos, 0, 2 * attn_span.item() - 1)
        rel_pos = rel_pos.to(self._rel_pos_embed.weight.device)
        return rel_pos

    def forward(self, seq_len: int) -> Tuple:
        """
        :param seq_len: length of sequence
        :return: Q, K w.r.t relative position embeddings. Shape [seq_len, seq_len, embed_dim]
        """
        attn_span = self._compute_attn_span(seq_len)
        c2p_rel_pos: torch.Tensor = self._get_rel_pos_idx(seq_len, attn_span, "c2p").unsqueeze(0)
        p2c_rel_pos: torch.Tensor = self._get_rel_pos_idx(seq_len, attn_span, "p2c").unsqueeze(0)

        # Clone() reason: https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html#torch.nn.Embedding
        if self._rel_pos_embed.max_norm is None:
            rel_embeds: torch.Tensor = self._rel_pos_embed.weight[self._max_rel_pos - attn_span:
                                                                  self._max_rel_pos + attn_span, :]
        else:
            rel_embeds: torch.Tensor = self._rel_pos_embed.weight[self._max_rel_pos - attn_span:
                                                                  self._max_rel_pos + attn_span, :].clone()
        # [seq_len, embed_dim] -> [1, seq_len, embed_dim]
        rel_embeds = rel_embeds.unsqueeze(0)
        rel_q, rel_k, _ = self._qkv(rel_embeds)
        return rel_q, rel_k, c2p_rel_pos, p2c_rel_pos
