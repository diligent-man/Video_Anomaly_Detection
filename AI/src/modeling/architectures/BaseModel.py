import gc
from copy import deepcopy
from functools import partial
from typing import List, Tuple, Any


import torch
from torch import Tensor
from torch.nn import Module, ModuleList


from ..necks import build_neck
from ..heads import build_head
from ..postprocessing import build_postprocessing
from ..backbones import build_backbone, ModelForwarder

from ...utils import DotDict
from .BaseModelOutput import BaseModelOutput


__all__ = ["BaseModel"]


class BaseModel(Module):
    def __init__(self, config: DotDict) -> None:
        super(BaseModel, self).__init__()
        config = config.Architecture

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

        config.neck["in_channels"] = out_channels
        neck, out_channels = build_neck(config)

        config.head["in_channels"] = out_channels
        head: Module = build_head(config)
        postprocessing: None | Module = build_postprocessing(config)

        self.__config: DotDict = config
        self.backbones: ModuleList = backbones
        self.__names: List[str] = names
        self._reduce: List[partial] = reduce
        self.out_proj: None | ModuleList = out_proj

        self.neck: Module = neck
        self.head: Module = head
        self.postprocessing: None | Module = postprocessing

        self.__return_extracted_feats = config.backbone.pop("return", False)
        self.__return_projected_feats = config.neck.pop("return", False)
        self.__return_neck_out = config.neck.pop("return", False)
        self.__return_dict = config.pop("return_dict", True)

    def forward(self, x: Tensor) -> BaseModelOutput | Tuple:
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

        device = x.device
        extracted_feats: None | List = None
        projected_feats: None | Tensor = None

        for i in range(len(self.backbones)):
            backbone: Module = self.backbones[i]
            name: str = self.__names[i]
            reduce: partial = self._reduce[i]

            with torch.autograd.inference_mode():
                feats: Tensor = ModelForwarder(backbone, name, reduce)(x.clone())

            feats = feats.clone()
            extracted_feats = [feats] if extracted_feats is None else extracted_feats.append(feats)

            if self.out_proj is not None:
                feats: Tensor = self.out_proj[i].to(device)(feats)

            feats = feats.unsqueeze(0)
            projected_feats = feats if projected_feats is None else torch.cat((projected_feats, feats), 0)

            # Clean cuda mem after forward 1 backbone
            self.backbones[i] = self.backbones[i].to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        neck_outs: Tensor = self.neck.to(device)(projected_feats)
        preds: Tensor = self.head.to(device)(neck_outs).squeeze(-1)  # (B, S, 1) -> (B, S)

        if self.postprocessing is not None:
            preds = self.postprocessing(preds)

        outs: BaseModelOutput = BaseModelOutput(
            preds=preds,
            extracted_feats=extracted_feats if self.__return_extracted_feats else None,
            projected_feats=projected_feats if self.__return_projected_feats else None,
            neck_outs=neck_outs if self.__return_neck_out else None
        )

        if not self.__return_dict:
            outs: Tuple[Any] = outs.to_tuple()
        return outs
