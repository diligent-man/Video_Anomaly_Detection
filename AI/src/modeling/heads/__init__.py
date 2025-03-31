from torch.nn import Module
from ...utils import DotDict
from .SimpleClassifier import SimpleClassifier

__all__ = ["build_head"]


heads = {
    "SimpleClassifier": SimpleClassifier,
}


def build_head(config: DotDict) -> Module:
    name = config.head.pop("name")
    assert name in heads.keys(), ValueError(f"Provided head is unavailable. Get '{name}'")

    in_channels = config.head.pop("in_channels")
    head: Module = heads[name](in_channels, **config.head.get_dict())
    return head
