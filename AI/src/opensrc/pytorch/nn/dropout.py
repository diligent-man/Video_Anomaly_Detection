from typing import Dict, Type
from torch.nn.modules import (
    Module,
    Dropout,
    Dropout1d,
    Dropout2d,
    Dropout3d,
    AlphaDropout,
    FeatureAlphaDropout
)


__all__ = ["avail_dropout"]


avail_dropout: Dict[str, Type[Module]] = {
    "Dropout": Dropout,
    "Dropout1d": Dropout1d,
    "Dropout2d": Dropout2d,
    "Dropout3d": Dropout3d,
    "AlphaDropout": AlphaDropout,
    "FeatureAlphaDropout": FeatureAlphaDropout
}
