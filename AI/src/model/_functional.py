import torch
from typing import Tuple


__all__ = ["compute_pad"]


@torch.fx.wrap
def compute_pad(kernel_size: int | Tuple[int, int, int],
                stride: int | Tuple[int, int, int],
                dim: int,
                dim_len: int
                ) -> int:
    pad = max(kernel_size[dim] - stride[dim], 0) if dim_len % stride[dim] == 0 else \
          max(kernel_size[dim] - (dim_len % stride[dim]), 0)
    return pad
