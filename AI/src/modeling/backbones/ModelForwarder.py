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
        assert x.dim() in (5, 6), ValueError("Input tensor should have dim 5 with shape (S, C, T, H, W) or (B, S, C, T, H, W)")

        if x.dim() == 5:
            x = x.unsqueeze(0)

        if self.__name in NET_2D:
            x: torch.Tensor = self._forward_2D_net(x)
        elif self.__name in NET_3D:
            x: torch.Tensor = self._forward_3D_net(x)
        return x

    def _forward_2D_net(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, T, H, W = x.shape

        # (B,S,C,T,H,W) -> (B*S,T,C,H,W) -> (B*S*T,C,H,W)
        try:
            tmp: torch.Tensor = x.view(-1, C, T, H, W).permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
            x: torch.Tensor = self.__model(tmp)
            x: torch.Tensor = _resolve_backbone_output(x)
            x = x.view(B * S, T, -1)
        except torch.OutOfMemoryError:
            cache: None | torch.Tensor = None

            for i in range(B):
                tmp: torch.Tensor = x[i, ...].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
                tmp: torch.Tensor = self.__model(tmp)
                tmp: torch.Tensor = _resolve_backbone_output(tmp)
                cache = tmp if cache is None else torch.cat((cache, tmp), dim=0)
            x = cache.view(B*S, T, -1)
        x = x.permute(0, -1, -2)  # (B*S,T,Hid_dim) -> # (B*S,Hid_dim,T)
        x = self.__reduce(**{"kernel_size": x.shape[-1]})(x)  # (B*S,Hid_dim,1)
        x = x.squeeze(-1).view(B, S, -1)  # (B*S,Hid_dim) -> (B,S,Hid_dim)
        return x

    def _forward_3D_net(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, T, H, W = x.shape
        x = x.view(-1, C, T, H, W)
        x: torch.Tensor = _resolve_backbone_output(self.__model(x))
        x = self.__reduce(**{"kernel_size": x.shape[2:]})(x)
        x = x.squeeze(dim=[2, 3, 4])  # (B,Hid_dim,T,H,W) -> (B, Hid_dim,1,1,1)
        x = x.view(B, S, -1)  # (B,S,Hid_dim)
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
