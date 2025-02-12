from typing import Dict, Any

import torch
from torchvision.models import WeightsEnum


__all__ = ["load_weights"]


def load_weights(weights: str | WeightsEnum) -> None | Dict[str, Any]:
    if isinstance(weights, WeightsEnum):
        try:
            weights = weights.get_state_dict(progress=True)
        except ValueError:
            weights = torch.load(weights.url, weights_only=True)
    elif isinstance(weights, str):
        weights = torch.load(weights, weights_only=True)
    return weights
