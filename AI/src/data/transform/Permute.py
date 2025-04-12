import gc
from typing import Tuple

import torch
from torchvision.transforms.v2 import Transform


__all__ = ["Permute"]


class Permute(Transform):
    """
    Permutes the dimensions of a video.
    """
    def __init__(self, dims: Tuple[int], device="cpu", clear_cuda_mem: bool = True):
        """
        :param dims (Tuple[int]): The desired ordering of dimensions.
        """
        assert (
            (d in dims) for d in range(len(dims))
        ), "dims must contain every dimension (0, 1, 2, ...)"

        super().__init__()
        self._dims = dims
        self._device = device
        self._clear_cuda_mem = clear_cuda_mem

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): video tensor whose dimensions are to be permuted.
        """
        x = x.to(self._device).permute(*self._dims)

        if self._clear_cuda_mem:
            gc.collect()
            torch.cuda.empty_cache()
        return x
