import functools
from typing import Dict, Any

import torch
import multipledispatch
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .constant import NET_2D, NET_3D


__all__ = ["ModelForwarder"]


class ModelForwarder(object):
    _model: torch.nn.Module
    _name: str
    _reduce: functools.partial

    def __init__(self, model: torch.nn.Module, name: str, reduce: functools.partial) -> None:
        super(ModelForwarder, self).__init__()
        self._model = model
        self._name = name
        self._reduce = reduce

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() in (5, 6), ValueError("Input tensor should have dim 5 with shape (S, C, T, H, W) or (B, S, C, T, H, W)")

        if x.dim() == 5:
            x = x.unsqueeze(0)

        if self._name in NET_2D:
            x: torch.Tensor = self._forward_2D_net(x)
        elif self._name in NET_3D:
            x: torch.Tensor = self._forward_3D_net(x)
        return x

    def _forward_2D_net(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, T, H, W = x.shape

        try:
            # (B,S,C,T,H,W) -> (B*S,T,C,H,W) -> (B*S*T,C,H,W)
            tmp: torch.Tensor = x.view(-1, C, T, H, W).permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
            x: torch.Tensor = _resolve_backbone_output(self._model.to(x.device)(tmp))
            x = x.view(B * S, T, -1)
        except torch.OutOfMemoryError:
            cache: None | torch.Tensor = None
            for i in range(B):
                tmp: torch.Tensor = x[i, ...].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
                tmp: torch.Tensor = _resolve_backbone_output(self._model.to(x.device)(tmp))
                cache = tmp if cache is None else torch.cat((cache, tmp), dim=0)
            x = cache.view(B*S, T, -1)
        # (B*S,T,Hid_dim) -> # (B*S,Hid_dim,T)
        x = x.permute(0, -1, -2)

        # (B*S,Hid_dim,1)
        x = self._reduce(**{"kernel_size": x.shape[-1]})(x)

        # (B*S,Hid_dim) -> (B,S,Hid_dim)
        x = x.squeeze(-1).view(B, S, -1)
        return x

    def _forward_3D_net(self, x: torch.Tensor) -> torch.Tensor:
        B, S, C, T, H, W = x.shape

        try:
            # (B,S,C,T,H,W) -> (B*S,T,C,H,W)
            tmp: torch.Tensor = x.view(-1, C, T, H, W)
            x: torch.Tensor = _resolve_backbone_output(self._model.to(x.device)(tmp))
        except torch.OutOfMemoryError:
            cache: None | torch.Tensor = None
            for i in range(B):
                tmp: torch.Tensor = x[i, ...]
                tmp: torch.Tensor = _resolve_backbone_output(self._model.to(x.device)(tmp))
                cache = tmp if cache is None else torch.cat((cache, tmp), dim=0)
            x = cache
        # (B*S,Hid_dim,T_out,H_out,W_out) -> (B*S,Hid_dim,1,1,1)
        x = self._reduce(**{"kernel_size": x.shape[2:]})(x)

        # (B,Hid_dim,T,H,W) -> (B, Hid_dim,1,1,1)
        x = x.squeeze(dim=[2, 3, 4])

        # (B,S,Hid_dim)
        x = x.view(B, S, -1)
        return x


@multipledispatch.dispatch(dict)
def _resolve_backbone_output(x: Dict[str, Any]) -> Any:
    if "features" in x.keys():
        x: torch.Tensor | Any = x["features"]
    elif isinstance(x, BaseModelOutputWithPooling):
        x: torch.Tensor = x[1]
    return _resolve_backbone_output(x)


@multipledispatch.dispatch(torch.Tensor)
def _resolve_backbone_output(x: torch.Tensor) -> torch.Tensor:
    return x
