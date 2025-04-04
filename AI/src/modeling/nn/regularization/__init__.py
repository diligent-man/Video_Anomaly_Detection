"""
This package is adapted from https://github.com/szymonmaszke/torchlayers
"""
from typing import Dict, Type

from torch.nn import Module

from .L1 import L1
from .L2 import L2


__all__ = ["avail_regularizers"]


avail_regularizers: Dict[str, Type[Module]] = {
    "L1": L1,
    "L2": L2
}
