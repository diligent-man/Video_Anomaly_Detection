import torch

from ...utils import DotDict
from .SimpleClassifier import SimpleClassifier

__all__ = ["build_head"]


heads = {
    "SimpleClassifier": SimpleClassifier,
}


def build_head(config: DotDict) -> torch.nn.Module:
    name = config.Architecture.head.pop("name")
    assert name in heads.keys(), ValueError(f"Provided head is unavailable. Get '{name}'")

    in_channels = config.Architecture.head.pop("in_channels")
    head: torch.nn.Module = heads[name](in_channels, **config.Architecture.head.get_dict())
    return head
