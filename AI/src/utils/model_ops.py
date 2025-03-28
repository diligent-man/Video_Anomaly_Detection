import os
import warnings
from typing import Any, Mapping, Tuple, Dict, List, Union

import torch
from torch.nn import Module
from torch.fx import GraphModule
from torchvision.models import WeightsEnum

from . import DotDict


__all__ = ["load_weights", "load_ckpt", "freeze_layer"]


def load_weights(weights: str | WeightsEnum, src: str = "pytorch", return_path: bool = False) -> str | Mapping[str, Any]:
    if isinstance(weights, WeightsEnum):
        # Consider path from this file
        rel_path: str = str(os.path.join(os.path.dirname(__file__), "..", "..", weights.url))

        if src == "hugging_face" or return_path:
            weights: str = rel_path
        else:
            if os.path.exists(rel_path):
                weights: Mapping[str, Any] = torch.load(rel_path, weights_only=True)
            else:
                weights: Mapping[str, Any] = weights.get_state_dict(progress=True)
    elif isinstance(weights, str):
        if src == "hugging_face" or return_path:
            weights: str = weights
        else:
            weights: Mapping[str, Any] = torch.load(weights, weights_only=True)
    return weights


def load_ckpt(config: DotDict,
              model: torch.nn.Module,
              optimizer: None | torch.optim.Optimizer,
              ) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
    """
    Implement after finish. training lop
    """
    load = config.Checkpoint.get("checkpoint", False)

    if load:
        ckpt_path = config.Checkpoint.get("resume_name", None)
        assert ckpt_path is not None, ValueError(f"Get '{ckpt_path}' for resume path while loading checkpoint")

        try:
            ckpt: Dict[str, torch.Tensor] = torch.load(ckpt_path, weights_only=True)
        except FileNotFoundError as e:
            ckpt = torch.hub.load_state_dict_from_url(ckpt_path)
            raise e

    # ckpt_cfg = config.Global.get("checkpoint", DotDict({}))
    # load: bool = config.Global.get("load", False)

    # if load:
    #     print("Loading checkpoint ...")
    return model, optimizer


def freeze_layer(model: Module | GraphModule,
                 freeze_list: int | List[str]
                 ) -> Tuple[Union[Module, GraphModule], int]:
    total_layers: int = len(list(model.parameters()))

    if isinstance(freeze_list, int) and freeze_list == -1:
        freeze_list = total_layers

    for i, (para_name, para) in enumerate(reversed(list(model.named_parameters()))):
        if (
                isinstance(freeze_list, int) and
                i in range(freeze_list)
        ) or (
                para_name in freeze_list
        ):
            para.requires_grad = False
    return model, total_layers
