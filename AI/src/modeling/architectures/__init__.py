import copy

import torch

from ...utils import DotDict
from .BaseModel import BaseModel
from .BaseModelOutput import BaseModelOutput


__all__ = ["build_model", "BaseModelOutput"]


def build_model(config: DotDict) -> torch.nn.Module:
    arch = BaseModel(copy.deepcopy(config))
    return arch
