from typing import Dict, Type

from torch.nn import Module

from .DistillInfoNCE import DistillInfoNCE
from .DistillBCEWithLogitsLoss import DistillBCEWithLogitsLoss


__all__ = ["avail_dl_loss"]


avail_dl_loss: Dict[str, Type[Module]] = {
    "DistillInfoNCE": DistillInfoNCE,
    "DistillBCEWithLogitsLoss": DistillBCEWithLogitsLoss
}
