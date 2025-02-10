from typing import List

import torch


__all__ = ["dynamic_expand", "transform_multihead"]


def dynamic_expand(x: torch.Tensor,
                   ref: torch.Tensor,
                   ref_dim: List[int]
                   ) -> torch.Tensor:
    """
    :param x: tensor to be expanded
    :param ref: referenced tensor
    :param ref_dim: dims of referenced tensor
    :return:
    """
    ref_dim = [ref.shape[dim] for dim in ref_dim]
    return x.expand(ref_dim)


def transform_multihead(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    """
    :param x: tensor of shape [batch_size, seq_len, embed_dim]
    :param num_heads: number of heads for splitting
    :return: multi-headed x. Shape [batch_size, num_heads, seq_len, head_dim]
    """
    batch, seq_len, embed_dim = x.shape
    head_dim = embed_dim // num_heads

    new_shape = (batch, seq_len, num_heads, head_dim)
    return x.view(new_shape).permute(0, 2, 1, 3)
