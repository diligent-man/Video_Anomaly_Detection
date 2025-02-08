import copy

from ...utils import DotDict
from .base_model import BaseModel


__all__ = ["build_model"]


def build_model(config: DotDict):
    arch = BaseModel(copy.deepcopy(config))
    return arch
