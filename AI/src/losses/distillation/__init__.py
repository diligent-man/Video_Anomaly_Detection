from typing import Dict, Type

from torch.nn import Module

from .DistillBCEWithLogitsLoss import DistillBCEWithLogitsLoss


__all__ = ["avail_dl_loss"]


avail_dl_loss: Dict[str, Type[Module]] = {
    "DistillBCEWithLogitsLoss": DistillBCEWithLogitsLoss
}
