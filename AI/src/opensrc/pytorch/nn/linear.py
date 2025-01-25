from torch.nn import (
    Bilinear,
    Identity,
    LazyLinear,
    Linear,
)

available_linear = {
    "Bilinear": Bilinear,
    "Identity": Identity,
    "LazyLinear": LazyLinear,
    "Linear": Linear
}