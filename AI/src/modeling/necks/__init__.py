from typing import Tuple

import torch

from ...utils import DotDict
from .TAN import TemporalAggregation


__all__ = ["build_neck"]


necks = {
    "TAN": TemporalAggregation
}


def build_neck(config: DotDict) -> Tuple[torch.nn.Module, int]:
    name = config.Architecture.neck.pop("name")
    assert name in necks.keys(), ValueError(f"Provided neck is unavailable. Get '{name}'")

    neck: torch.nn.Module = necks[name](**config.Architecture.neck.get_dict())
    out_channels: int = neck.out_channels
    return neck, out_channels
