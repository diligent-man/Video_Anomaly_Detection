from typing import Tuple

from torch.nn import Module

from ...utils import DotDict
from .TAN import TemporalAggregation


__all__ = ["build_neck", "TemporalAggregation"]


NECKS = {
    "TAN": TemporalAggregation
}


def build_neck(config: DotDict) -> Tuple[Module, int]:
    name = config.Architecture.neck.pop("name")
    assert name in NECKS.keys(), ValueError(f"Provided neck is unavailable. Get '{name}'")

    neck: Module = NECKS[name](**config.Architecture.neck.get_dict())
    out_channels: int = neck.out_channels
    return neck, out_channels
