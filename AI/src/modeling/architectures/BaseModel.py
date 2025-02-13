import functools
from typing import List, Tuple, Any

import torch

from ..necks import build_neck
from ..heads import build_head
from ..postprocessing import build_postprocessing
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

        # backbone, neck, head need to be configured
        backbones, names, reduce, out_proj, out_channels = build_backbone(config)

        config.Architecture.neck["in_channels"] = out_channels
        neck, out_channels = build_neck(config)

        config.Architecture.head["in_channels"] = out_channels
        head: torch.nn.Module = build_head(config)

        postprocessing: None | torch.nn.Module = build_postprocessing(config)
        self.__config: DotDict = config

        self._backbones: torch.nn.ModuleList = backbones
        self._names: List[str] = names
        self._reduce: List[functools.partial] = reduce
        self._out_proj: None | torch.nn.ModuleList = out_proj

        self._neck: torch.nn.Module = neck
        self._head: torch.nn.Module = head
        self._postprocessing: None | torch.nn.Module = postprocessing

        self._return_extracted_feats = config.Architecture.backbone.pop("return", False)
        self._return_projected_feats = config.Architecture.neck.pop("return", False)
        self._return_neck_out = config.Architecture.neck.pop("return", False)
        self._return_dict = config.Architecture.pop("return_dict", True)

    def forward(self, x: torch.Tensor) -> BaseModelOutput | Tuple:
        """
        :param x: list of input tensors for corresponding backbones.
                  Shape (S,C,T,H,W) or (B,S,C,T,H,W)
        :return: BaseModelOutput obj
        """
        assert x.dim() in (5, 6), ValueError(
            "Input tensor should have dim 5 with shape (S, C, T, H, W) or (B, S, C, T, H, W)"
        )

        if x.dim() == 5:
            x = x.unsqueeze(0)

        extracted_feats: None | List = None
        projected_feats: None | torch.Tensor = None
        for i in range(len(self._backbones)):
            backbone: torch.nn.Module = self._backbones[i].to(x.device)
            name: str = self._names[i]
            reduce: functools.partial = self._reduce[i]

            feats: torch.Tensor = ModelForwarder(backbone, name, reduce)(x.clone())
            extracted_feats = [feats] if extracted_feats is None else extracted_feats.append(feats)

            if self._out_proj is not None:
                feats: torch.Tensor = self._out_proj[i].to(feats.device)(feats)

            feats = feats.unsqueeze(0)
            projected_feats = feats if projected_feats is None else torch.cat((projected_feats, feats), 0)

        neck_outs: torch.Tensor = self._neck.to(projected_feats.device)(projected_feats)
        preds: torch.Tensor = self._head.to(neck_outs.device)(neck_outs)

        if self._postprocessing is not None:
            preds = self._postprocessing(preds)

        outs: BaseModelOutput = BaseModelOutput(
            preds=preds,
            extracted_feats=extracted_feats if self._return_extracted_feats else None,
            projected_feats=projected_feats if self._return_projected_feats else None,
            neck_outs=neck_outs if self._return_neck_out else None
        )

        if not self._return_dict:
            outs: Tuple[Any] = outs.to_tuple()
        return outs
