import gc
from functools import partial
from typing import List, Tuple, Any


import torch
from torch import Tensor
from torch.nn import Module, ModuleList


from ..backbones import build_backbone, MultiBackboneForwarder
from ..necks import build_neck
from ..heads import build_head, SimpleClassifierOutput

from ...utils import DotDict
from .ModelOutput import BaseModelOutput
from ..postprocessing import build_postprocessing


__all__ = ["BaseModel"]


class BaseModel(Module):
    def __init__(self, config: DotDict) -> None:
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
        return_extracted_feats = config.backbone.pop("return_extracted_feats", False)
        return_projected_feats = config.backbone.pop("return_projected_feats", False)
        backbones, names, reduce, out_proj, out_channels = build_backbone(config)

        return_neck_out = config.neck.pop("return_neck_out", False)
        config.neck["in_channels"] = out_channels
        neck, out_channels = build_neck(config)

        config.head["in_channels"] = out_channels
        head, return_logits = build_head(config)
        postprocessing: None | Module = build_postprocessing(config)

        self.__config: DotDict = config
        self.backbones: ModuleList = backbones
        self.__names: List[str] = names
        self._reduce: List[partial] = reduce
        self.out_proj: None | ModuleList = out_proj

        self.neck: Module = neck
        self.head: Module = head
        self.postprocessing: None | Module = postprocessing

        self.__return_extracted_feats: bool = return_extracted_feats
        self.__return_projected_feats: bool = return_projected_feats
        self.__return_neck_out: bool = return_neck_out
        self.__return_logits: bool = return_logits
        self.__return_dict: bool = config.pop("return_dict", True)

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
        extracted_feats: None | List[Tensor] = None
        projected_feats: None | Tensor = None

        # Feed backbone
        for i in range(len(self.backbones)):
            backbone: Module = self.backbones[i]
            name: str = self.__names[i]
            reduce: partial = self._reduce[i]

            feats: Tensor = MultiBackboneForwarder(backbone, name, reduce)(x)
            extracted_feats = [feats] if extracted_feats is None else extracted_feats.append(feats)

            if self.out_proj is not None:
                feats: Tensor = self.out_proj[i].to(device)(feats)

            feats = feats.unsqueeze(0)
            projected_feats = feats if projected_feats is None else torch.cat((projected_feats, feats), 0)

            self.backbones[i] = self.backbones[i].to("cpu")
            gc.collect()
            torch.cuda.empty_cache()

        # Feed neck
        neck_outs: Tensor = self.neck.to(device)(projected_feats)  # (B, S, Hid_dim)

        # Feed head
        # preds, logits (if have): (B, S, Hid_dim) -> (B, S)
        head_outs: SimpleClassifierOutput = self.head.to(device)(neck_outs)

        if self.postprocessing is not None:
            head_outs.preds = self.postprocessing(head_outs.preds)

        outs: BaseModelOutput = BaseModelOutput(
            extracted_feats=extracted_feats if self.__return_extracted_feats else None,
            projected_feats=projected_feats if self.__return_projected_feats else None,
            neck_outs=neck_outs if self.__return_neck_out else None,
            logits=head_outs.logits,
            preds=head_outs.preds
        )

        if not self.__return_dict:
            outs: Tuple[Any] = outs.to_tuple()
        return outs
