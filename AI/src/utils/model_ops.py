import os
from typing import Any, Mapping, Tuple, List, Union

import torch
from torch.nn import Module
from torch.fx import GraphModule
from torchvision.models import WeightsEnum

__all__ = ["load_weights", "freeze_layer"]


def load_weights(weights: str | WeightsEnum,
                 src: str = "pytorch",
                 return_path: bool = False
                 ) -> str | Mapping[str, Any]:
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


def freeze_layer(model: Module | GraphModule,
                 trainable_layers: int | List[str]
                 ) -> Tuple[Union[Module, GraphModule], int, List[str]]:
    total_layers: int = len(list(model.parameters()))
    trainable_layer_names: List[str] = []

    if isinstance(trainable_layers, int):
        trainable_layers = total_layers if trainable_layers == -1 else trainable_layers

    for i, (para_name, para) in enumerate(reversed(list(model.named_parameters()))):
        if isinstance(trainable_layers, int):
            if i not in range(trainable_layers):
                para.requires_grad = False
        else:
            if para_name not in trainable_layers:
                para.requires_grad = False

        if para.requires_grad:
            trainable_layer_names.append(para_name)
    return model, total_layers, trainable_layer_names
