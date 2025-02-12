import copy
import functools
from typing import List, Any, Dict

import torch

from ..necks import build_neck
from ..heads import build_head
from ..backbones import build_backbone, ModelForwarder

from ...utils import DotDict
from .BaseModelOutput import BaseModelOutput

__all__ = ["BaseModel"]


class BaseModel(torch.nn.Module):
    def __init__(self, config: DotDict):
        super(BaseModel, self).__init__()

        # build transform,
        # if "Transform" not in config or config["Transform"] is None:
        #     self.use_transform = False
        # else:
        #     self.use_transform = True
        #     config["Transform"]["in_channels"] = in_channels
        #     self.transform = build_transform(config["Transform"])
        #     in_channels = self.transform.out_channels

        self._return_backbone_feats = config.Architecture.backbone.pop("return_backbone_feats", False)
        self._return_neck_feats = config.Architecture.neck.pop("return_backbone_feats", False)

        # backbone, neck, head must need to be configured
        backbones, names, reduce, out_proj, out_channels = build_backbone(config)

        config.Architecture.neck["in_channels"] = out_channels
        neck, out_channels = build_neck(config)

        config.Architecture.head["in_channels"] = out_channels
        head = build_head(config)

        self.__config: DotDict = config

        self._backbones: torch.nn.ModuleList = backbones
        self._names: List[str] = names
        self._reduce: List[functools.partial] = reduce
        self._out_proj: None | torch.nn.ModuleList = out_proj

        self._neck: torch.nn.Module = neck
        self._head: torch.nn.Module = head

    def forward(self, x: torch.Tensor) -> BaseModelOutput:
        """
        :param x: list of input tensors for corresponding backbones.
                  Shape (S,C,T,H,W) or (B,S,C,T,H,W)
        :return:
        """
        assert x.dim() in (5, 6), ValueError(
            "Input tensor should have dim 5 with shape (S, C, T, H, W) or (B, S, C, T, H, W)"
        )

        if x.dim() == 5:
            x = x.unsqueeze(0)

        # B, C, T, H, W = x.shape

        # Rule-based approach
        # backbone_out: None | torch.Tensor = None

        for i in range(len(self._backbones)):
            backbone: torch.nn.Module = self._backbones[i].to(x.device)
            name: str = self._names[i]
            reduce: functools.partial = self._reduce[i]

            feats: torch.Tensor = ModelForwarder(backbone, name, reduce)(x.clone())

            if self._out_proj is not None:
                feats: torch.Tensor = self._out_proj[i].to(feats.device)(feats)

        # print(self._neck)

        # return BaseModelOutput(
        #     preds,
        #     backbone_out if self._return_backbone_feats else None,
        #     neck_out if self._return_backbone_feats else None
        # )

        # print(backbone_out.shape)
        # neck_out = self._neck(backbone_out)
        # preds = self._head(neck_out)
