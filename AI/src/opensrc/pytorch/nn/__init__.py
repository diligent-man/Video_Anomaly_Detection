from .loss import available_loss
from .conv import available_conv
from .linear import available_linear
from .dropout import available_dropout
from .flatten import available_flatten
from .pooling import available_pooling
from .activation import available_activation
from .normalization import available_normalization

__all__ = ["available_loss", "available_layer"]


available_layer = {
    **available_conv,
    **available_linear,
    **available_dropout,
    **available_flatten,
    **available_pooling,
    **available_activation,
    **available_normalization
}
