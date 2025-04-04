from typing import Dict, Type
from torch.nn.modules import (
    Module,
    Flatten,
    Unflatten
)


__all__ = ["avail_flatten"]


avail_flatten: Dict[str, Type[Module]] = {
    "Flatten": Flatten,
    "Unflatten": Unflatten
}
