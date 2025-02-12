import os.path
from typing import Any, Mapping

import torch
from torchvision.models import WeightsEnum


__all__ = ["load_weights"]


def load_weights(weights: str | WeightsEnum) -> Mapping[str, Any]:
    if isinstance(weights, WeightsEnum):
        if os.path.exists(weights.url):
            weights: Mapping[str, Any] = torch.load(weights.url, weights_only=True)
        else:
            weights: Mapping[str, Any] = weights.get_state_dict(progress=True)
    elif isinstance(weights, str):
        weights: Mapping[str, Any] = torch.load(weights, weights_only=True)
    return weights
