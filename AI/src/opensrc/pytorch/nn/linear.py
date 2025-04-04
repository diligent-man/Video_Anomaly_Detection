from typing import Dict, Type
from torch.nn.modules import (
    Module,
    Bilinear,
    Identity,
    LazyLinear,
    Linear,
)


__all__ = ["avail_linear"]


avail_linear: Dict[str, Type[Module]] = {
    "Bilinear": Bilinear,
    "Identity": Identity,
    "LazyLinear": LazyLinear,
    "Linear": Linear
}
