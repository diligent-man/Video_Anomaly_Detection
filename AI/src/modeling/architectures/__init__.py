from torch.nn import Module

from ...utils import DotDict
from .BaseModel import BaseModel
from .BaseModelOutput import BaseModelOutput
from .VADDistillationModel import VADDistillationModel

__all__ = [
    "build_model",

    "BaseModel",
    "VADDistillationModel",

    "BaseModelOutput"
]


def build_model(config: DotDict) -> Module:
    algorithm: str = config.Architecture.pop("algorithm", "single")
    if algorithm == "single":
        arch = BaseModel(config.Architecture)
    else:
        arch = VADDistillationModel(config)
    # exit()
    return arch
