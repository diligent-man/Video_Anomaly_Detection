from typing import Union, Tuple

import torch

from ..functional import compute_pad


__all__ = ["MaxPool3dSamePadding"]


class MaxPool3dSamePadding(torch.nn.Module):
    _is_leaf_module = True
    _kernel_size: Union[int, Tuple[int, int, int]]
    _stride: Union[int, Tuple[int, int, int]]
    _padding: Union[int, Tuple[int, int, int]]

    def __init__(self,
                 kernel_size: Union[int, Tuple[int, int, int]] = (3, 3, 3),
                 stride: Union[int, Tuple[int, int, int]] = (1, 1, 1),
                 padding: Union[int, Tuple[int, int, int]] = 0,
                 ) -> None:
        super(MaxPool3dSamePadding, self).__init__()
        self._kernel_size = kernel_size
        self._stride = stride
        self._padding = padding

    @property
    def is_leaf_module(self):
        return self._is_leaf_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (batch, channel, t, h, w) = x.size()

        pad_t: int = compute_pad(self._kernel_size, self._stride, 0, t)
        pad_h: int = compute_pad(self._kernel_size, self._stride, 1, h)
        pad_w: int = compute_pad(self._kernel_size, self._stride, 2, w)

        pad_t_f: int = pad_t // 2
        pad_t_b: int = pad_t - pad_t_f

        pad_h_f: int = pad_h // 2
        pad_h_b: int = pad_h - pad_h_f

        pad_w_f: int = pad_w // 2
        pad_w_b: int = pad_w - pad_w_f

        pad: Tuple[int, int, int, int, int, int] = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)

        x: torch.Tensor = torch.nn.functional.pad(x, pad)
        return torch.nn.MaxPool3d(self._kernel_size, self._stride, self._padding)(x)
