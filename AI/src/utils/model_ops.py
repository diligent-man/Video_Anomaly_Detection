import os.path
from typing import Any, Mapping, Tuple

import torch
from torchvision.models import WeightsEnum

from . import DotDict


__all__ = ["load_weights", "load_ckpt"]


def load_weights(weights: str | WeightsEnum) -> Mapping[str, Any]:
    if isinstance(weights, WeightsEnum):
        if os.path.exists(weights.url):
            weights: Mapping[str, Any] = torch.load(weights.url, weights_only=True)
        else:
            weights: Mapping[str, Any] = weights.get_state_dict(progress=True)
    elif isinstance(weights, str):
        weights: Mapping[str, Any] = torch.load(weights, weights_only=True)
    return weights


def load_ckpt(config: DotDict,
              model: torch.nn.Module,
              optimizer: None | torch.optim.Optimizer,
              ) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    """
    Implement after finish. training lop
    """
    config.Global.get("checkpoint", DotDict({}))
    # ckpt_cfg = config.Global.get("checkpoint", DotDict({}))
    # load: bool = config.Global.get("load", False)

    # if load:
    #     print("Loading checkpoint ...")
    return model, optimizer
