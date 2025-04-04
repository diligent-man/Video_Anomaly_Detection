from typing import Dict
from torch.nn.modules import (
    Module,
    Conv1d,
    Conv2d,
    Conv3d,
    ConvTranspose1d,
    ConvTranspose2d,
    ConvTranspose3d,
    LazyConv1d,
    LazyConv2d,
    LazyConv3d,
    LazyConvTranspose1d,
    LazyConvTranspose2d,
    LazyConvTranspose3d
)


__all__ = ["avail_conv"]


avail_conv: Dict[str, Module] = {
    "Conv1d": Conv1d,
    "Conv2d": Conv2d,
    "Conv3d": Conv3d,
    "ConvTranspose1d": ConvTranspose1d,
    "ConvTranspose2d": ConvTranspose2d,
    "ConvTranspose3d": ConvTranspose3d,
    "LazyConv1d": LazyConv1d,
    "LazyConv2d": LazyConv2d,
    "LazyConv3d": LazyConv3d,
    "LazyConvTranspose1d": LazyConvTranspose1d,
    "LazyConvTranspose2d": LazyConvTranspose2d,
    "LazyConvTranspose3d": LazyConvTranspose3d
}
