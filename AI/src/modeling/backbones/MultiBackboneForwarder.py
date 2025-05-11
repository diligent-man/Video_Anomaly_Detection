import warnings
from typing import Dict, Any
from functools import partial

import torch
from torch import Tensor
from torch.nn import Module

import multipledispatch
from transformers.modeling_outputs import BaseModelOutputWithPooling


from .constant import NET_2D, NET_3D


__all__ = ["MultiBackboneForwarder"]


class MultiBackboneForwarder(object):
    _model: Module
    _name: str
    _reduce: partial

    def __init__(self, model: Module, name: str, reduce: partial) -> None:
        super(MultiBackboneForwarder, self).__init__()
        self._model = model
        self._name = name
        self._reduce = reduce

    def __call__(self, x: Tensor) -> Tensor:
        assert x.dim() in (5, 6), ValueError("Input tensor should have dim 5 with shape (S, C, T, H, W) or (B, S, C, T, H, W)")

        if x.dim() == 5:
            x = x.unsqueeze(0)

        self._model = self._model.to(x.device)

        if self._name in NET_2D:
            x: Tensor = self._forward_2D_net(x)
        elif self._name in NET_3D:
            x: Tensor = self._forward_3D_net(x)
        return x

    def _forward_2D_net(self, x: Tensor) -> Tensor:
        B, S, C, T, H, W = x.shape

        try:
            # (B,S,C,T,H,W) -> (B*S,T,C,H,W) -> (B*S*T,C,H,W)
            tmp: Tensor = x.view(-1, C, T, H, W).permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)

            x: Tensor = _resolve_backbone_output(self._model(tmp))
            x = x.view(B * S, T, -1)
        except torch.OutOfMemoryError:
            batch_cache: None | Tensor = None
            for i in range(B):
                try:
                    batch: Tensor = x[i, ...].permute(0, 2, 1, 3, 4).reshape(-1, C, H, W)
                    batch: Tensor = _resolve_backbone_output(self._model(batch))
                    batch_cache = batch if batch_cache is None else torch.cat((batch_cache, batch), dim=0)
                except torch.OutOfMemoryError as e:
                    snippet_cache: None | Tensor = None
                    for j in range(S):
                        snippet: Tensor = x[i, j, ...].permute(1, 0, 2, 3)
                        snippet: Tensor = _resolve_backbone_output(self._model(snippet))

                        snippet_cache = snippet if snippet_cache is None else torch.cat((snippet_cache, snippet), dim=0)
                    batch_cache = snippet_cache if batch_cache is None else torch.cat((batch_cache, snippet_cache), dim=0)

            x = batch_cache.view(B*S, T, -1)
        # (B*S,T,Hid_dim) -> (B*S,Hid_dim,T)
        x = x.permute(0, -1, -2)

        # (B*S,Hid_dim,T) -> (B*S,Hid_dim,1)
        if self._reduce.func is torch.nn.AvgPool1d:
            x = self._reduce(**{"kernel_size": x.shape[-1]})(x)
        else:
            x = self._reduce(x, 1, 2)

        # (B*S,Hid_dim,1) -> (B,S,Hid_dim) -> (B,S,Hid_dim)
        x = x.squeeze(-1).view(B, S, -1)
        return x

    def _forward_3D_net(self, x: Tensor) -> Tensor:
        B, S, C, T, H, W = x.shape

        try:
            # (B,S,C,T,H,W) -> (B*S,T,C,H,W)
            tmp: Tensor = x.view(-1, C, T, H, W)
            x: Tensor = _resolve_backbone_output(self._model(tmp))
        except torch.OutOfMemoryError as e:
            cache: None | Tensor = None
            for i in range(B):
                tmp: Tensor = x[i, ...]
                tmp: Tensor = _resolve_backbone_output(self._model(tmp))
                cache = tmp if cache is None else torch.cat((cache, tmp), dim=0)
            x = cache

        # (B*S,Hid_dim,T_out,H_out,W_out) -> (B*S,Hid_dim,1,1,1)
        if self._reduce.func is torch.nn.AvgPool3d:
            x = self._reduce(**{"kernel_size": x.shape[2:]})(x)
        else:
            x = self._reduce(x, 1, 2)

        # (B,Hid_dim,1,1,1) -> (B, Hid_dim)
        x = x.squeeze(dim=[2, 3, 4])

        # (B,S,Hid_dim)
        x = x.view(B, S, -1)
        return x


@multipledispatch.dispatch(dict)
def _resolve_backbone_output(x: Dict[str, Any]) -> Any:
    if "features" in x.keys():
        x: Tensor | Any = x["features"]
    elif isinstance(x, BaseModelOutputWithPooling):
        x: Tensor = x[1]
    return _resolve_backbone_output(x)


@multipledispatch.dispatch(Tensor)
def _resolve_backbone_output(x: Tensor) -> Tensor:
    return x
