from copy import deepcopy
from torch.nn import Module

from ...utils import DotDict
from .BaseModel import BaseModel
from .BaseModelOutput import BaseModelOutput
from .DistillationModel import DistillationModel

__all__ = ["build_model", "BaseModelOutput"]


def build_model(config: DotDict) -> Module:
    algorithm: str = config.Architecture.pop("algorithm", "single")
    if algorithm == "single":
        arch = BaseModel(config)
    else:
        arch = DistillationModel(config)
    # exit()
    return arch
