import functools
from typing import Dict, Any

import torch
import multipledispatch
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .constant import NET_2D, NET_3D


__all__ = ["ModelForwarder"]


class ModelForwarder(torch.nn.Module):
    __model: torch.nn.Module
    __name: str
    __reduce: functools.partial

    def __init__(self, model: torch.nn.Module, name: str, reduce: functools.partial) -> None:
        super(ModelForwarder, self).__init__()
        self.__model = model
        self.__name = name
        self.__reduce = reduce

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.__name in NET_2D:
            x: torch.Tensor = self._forward_2D_net(x)
        elif self.__name in NET_3D:
            x: torch.Tensor = self._forward_3D_net(x)
        return x

    def _forward_2D_net(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5, ValueError("Input tensor should have dim 5 with shape (B, C, T, H, W)")
        B, C, T, H, W = x.shape

        # (B, C, T, H, W) -> (B, T, C, H, W) -> (-1, C, H, W)
        x = x.permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
        x = _resolve_backbone_output(self.__model(x))
        x = x.view(B, T, -1).permute(0, -1, 1)  # (B, T, Hid_dim) -> # (B, Hid_dim, T)
        x = self.__reduce(**{"kernel_size": x.shape[-1]})(x)  # (B, Hid_dim, 1)
        x = x.squeeze(-1)
        return x

    def _forward_3D_net(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() == 5, ValueError("Input tensor should have dim 5 with shape (B, C, T, H, W)")
        x = _resolve_backbone_output(self.__model(x))
        x = self.__reduce(**{"kernel_size": x.shape[2:]})(x)
        x = x.squeeze(dim=[2, 3, 4])  # (B, Hid_dim, T, H, W) -> (B, Hid_dim, 1, 1, 1)
        return x


@multipledispatch.dispatch(dict)
def _resolve_backbone_output(x: Dict[str, Any]) -> Any:
    if "features" in x.keys():
        x: torch.Tensor | Any = x["features"]
    elif isinstance(x, BaseModelOutputWithPooling):
        x: torch.Tensor = x[1]
    return _resolve_backbone_output(x)


@multipledispatch.dispatch(torch.Tensor)
def _resolve_backbone_output(x: torch.Tensor) -> Any:
    return x
