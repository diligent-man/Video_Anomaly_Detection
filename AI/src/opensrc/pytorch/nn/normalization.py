# batchnorm
from torch.nn import (
    BatchNorm1d,
    LazyBatchNorm1d,
    BatchNorm2d,
    LazyBatchNorm2d,
    BatchNorm3d,
    LazyBatchNorm3d,
    SyncBatchNorm
)

# instancenorm
from torch.nn import (
    InstanceNorm1d,
    InstanceNorm2d,
    InstanceNorm3d,
    LazyInstanceNorm1d,
    LazyInstanceNorm2d,
    LazyInstanceNorm3d
)

# normalization
from torch.nn import (
    LocalResponseNorm,
    CrossMapLRN2d,
    LayerNorm,
    GroupNorm
)


available_normalization = {
    # batchnorm
    "BatchNorm1d": BatchNorm1d,
    "LazyBatchNorm1d": LazyBatchNorm1d,
    "BatchNorm2d": BatchNorm2d,
    "LazyBatchNorm2d": LazyBatchNorm2d,
    "BatchNorm3d": BatchNorm3d,
    "LazyBatchNorm3d": LazyBatchNorm3d,
    "SyncBatchNor": SyncBatchNorm,

    # instancenorm
    "InstanceNorm1d": InstanceNorm1d,
    "InstanceNorm2d": InstanceNorm2d,
    "InstanceNorm3d": InstanceNorm3d,
    "LazyInstanceNorm1d": LazyInstanceNorm1d,
    "LazyInstanceNorm2d": LazyInstanceNorm2d,
    "LazyInstanceNorm3": LazyInstanceNorm3d,
                                          
    # normalization
    "LocalResponseNorm": LocalResponseNorm,
    "CrossMapLRN2d": CrossMapLRN2d,
    "LayerNorm": LayerNorm,
    "GroupNorm": GroupNorm
}