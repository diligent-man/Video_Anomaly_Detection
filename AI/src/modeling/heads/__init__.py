from typing import Tuple

from torch.nn import Module

from ...utils import DotDict
from .SimpleClassifier import SimpleClassifier
from .HeadOutput import SimpleClassifierOutput

__all__ = ["build_head", "SimpleClassifierOutput"]


heads = {
    "SimpleClassifier": SimpleClassifier,
}


def build_head(config: DotDict) -> Tuple[Module, bool]:
    name = config.head.pop("name")
    assert name in heads.keys(), ValueError(f"Provided head is unavailable. Get '{name}'")

    return_logits: bool = config.head.get("return_logits", False)
    head: Module = heads[name](**config.head.get_dict())
    return head, return_logits
