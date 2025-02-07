from torch.nn import (
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

available_conv = {
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
