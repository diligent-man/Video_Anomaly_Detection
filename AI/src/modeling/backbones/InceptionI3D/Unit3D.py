from typing import Tuple, Union, Callable

import torch

from ..functional import compute_pad


__all__ = ["Unit3D"]


class Unit3D(torch.nn.Module):
    _is_leaf_module = True
    in_channels: int
    output_channels: int
    kernel_size: Union[int, Tuple[int, int, int]]
    stride: Union[int, Tuple[int, int, int]]
    padding: Union[int, Tuple[int, int, int]]
    activation_fn: Callable
    use_batch_norm: bool
    use_bias: bool
    name: str

    def __init__(self,
                 in_channels: int,
                 output_channels: int,
                 kernel_size: Union[int, Tuple[int, int, int]] = (1, 1, 1),
                 stride: Union[int, Tuple[int, int, int]] = (1, 1, 1),
                 padding: Union[int, Tuple[int, int, int]] = (0, 0, 0),
                 activation_fn: Union[None, Callable] = torch.nn.functional.relu,
                 use_batch_norm: bool = True,
                 use_bias: bool = False,
                 name: str = "Unit3D"
                 ) -> None:

        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()
        self._output_channels = output_channels
        self._kernel_size = kernel_size
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding

        self.conv3d = torch.nn.Conv3d(in_channels,
                                      self._output_channels,
                                      self._kernel_size,
                                      self._stride,
                                      0,  # we always want padding to be 0 here. We will dynamically pad based on input size in forward function
                                      bias=self._use_bias)

        if self._use_batch_norm:
            self.batch3d = torch.nn.BatchNorm3d(self._output_channels, eps=0.001, momentum=0.01)

    @property
    def is_leaf_module(self):
        return self._is_leaf_module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        (batch, channel, t, h, w) = x.size()

        pad_t = compute_pad(self._kernel_size, self._stride, 0, t)
        pad_h = compute_pad(self._kernel_size, self._stride, 1, h)
        pad_w = compute_pad(self._kernel_size, self._stride, 2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f

        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f

        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = (pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b)

        x: torch.Tensor = torch.nn.functional.pad(x, pad)
        x: torch.Tensor = self.conv3d(x)

        if self._use_batch_norm:
            x: torch.Tensor = self.batch3d(x)

        if self._activation_fn is not None:
            x: torch.Tensor = self._activation_fn(x)
        return x