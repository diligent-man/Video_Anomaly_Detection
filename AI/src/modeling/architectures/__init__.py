from torch.nn import Module


from .ModelOutput import (
    BaseModelOutput,
    VADDistillModelOutput
)

from ...utils import DotDict
from .BaseModel import BaseModel
from .VADDistillationModel import VADDistillationModel


__all__ = [
    "build_model",

    "BaseModel",
    "BaseModelOutput",

    "VADDistillationModel",
    "VADDistillModelOutput"
]


def build_model(config: DotDict) -> Module:
    algorithm: str = config.Architecture.pop("algorithm", "single")
    if algorithm == "single":
        arch = BaseModel(config.Architecture)
    else:
        arch = VADDistillationModel(config)
    return arch
