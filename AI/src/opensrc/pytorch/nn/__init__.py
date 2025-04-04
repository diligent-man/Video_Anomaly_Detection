from typing import Dict

from torch.nn import Module

from .loss import avail_loss
from .conv import avail_conv
from .linear import avail_linear
from .dropout import avail_dropout
from .flatten import avail_flatten
from .pooling import avail_pooling
from .activation import avail_act
from .normalization import avail_norm


__all__ = [
    "avail_loss",
    "avail_conv",
    "avail_linear",
    "avail_dropout",
    "avail_flatten",
    "avail_pooling",
    "avail_act",
    "avail_norm",
    "avail_layer"
]


avail_layer: Dict[str, Module] = {
    **avail_conv,
    **avail_linear,
    **avail_dropout,
    **avail_flatten,
    **avail_pooling,
    **avail_act,
    **avail_norm
}
