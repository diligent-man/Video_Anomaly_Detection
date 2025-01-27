from typing import Tuple, List

import torch


__all__ = ["compute_pad", "dynamic_expand"]


def compute_pad(kernel_size: int | Tuple[int, int, int],
                stride: int | Tuple[int, int, int],
                dim: int,
                dim_len: int
                ) -> int:
    pad = max(kernel_size[dim] - stride[dim], 0) if dim_len % stride[dim] == 0 else \
          max(kernel_size[dim] - (dim_len % stride[dim]), 0)
    return pad


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
