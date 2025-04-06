from torch.nn import Module

from .ModelOutput import (
    ModelOutput,
    BaseModelOutput,
    VADDistillModelOutput
)

from ...utils import DotDict
from .BaseModel import BaseModel
from .VADDistillModel import VADDistillModel


__all__ = [
    "build_model",
    "ModelOutput",

    "BaseModel",
    "BaseModelOutput",

    "VADDistillModel",
    "VADDistillModelOutput"
]


def build_model(config: DotDict) -> Module:
    algorithm: str = config.Architecture.pop("algorithm", "single")
    if algorithm == "single":
        arch = BaseModel(config.Architecture)
    else:
        arch = VADDistillModel(config)
    return arch
