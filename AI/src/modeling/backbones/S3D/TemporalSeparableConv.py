from typing import Callable

import torch
from torchvision.ops import Conv3dNormActivation


__all__ = ["TemporalSeparableConv"]


class TemporalSeparableConv(torch.nn.Sequential):
    def __init__(
        self,
        in_planes: int,
        out_planes: int,
        kernel_size: int,
        stride: int,
        padding: int,
        norm_layer: Callable[..., torch.nn.Module],
    ):
        super().__init__(
            Conv3dNormActivation(
                in_planes,
                out_planes,
                kernel_size=(1, kernel_size, kernel_size),
                stride=(1, stride, stride),
                padding=(0, padding, padding),
                bias=False,
                norm_layer=norm_layer,
            ),
            Conv3dNormActivation(
                out_planes,
                out_planes,
                kernel_size=(kernel_size, 1, 1),
                stride=(stride, 1, 1),
                padding=(padding, 0, 0),
                bias=False,
                norm_layer=norm_layer,
            ),
        )
