import copy
from typing import List, Dict, Any

import torch

from  multipledispatch import dispatch
from transformers.modeling_outputs import BaseModelOutputWithPooling

from ...utils import DotDict
from ..necks import build_neck
from ..heads import build_head
from ..backbones import build_backbone
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
        backbones, backbone_names, out_proj, out_channels = build_backbone(copy.deepcopy(config))

        config.Architecture.neck["in_channels"] = out_channels
        neck, out_channels = build_neck(copy.deepcopy(config))

        config.Architecture.head["in_channels"] = out_channels
        head = build_head(copy.deepcopy(config))

        self._backbones: torch.nn.ModuleList = backbones
        self._backbone_names: List[str] = backbone_names
        self._neck: torch.nn.Module = neck
        self._head: torch.nn.Module = head
        self._out_proj: None | torch.nn.ModuleList = out_proj
        self._reduce: str = config.Architecture.backbone.get("reduce", "mean")

    def forward(self, x: torch.Tensor) -> BaseModelOutput:
        """
        :param x: list of input tensors for corresponding backbones. Shape (B, C, T, H, W)
        :return:
        """
        B, C, T, H, W = x.shape

        # Rule-based approach
        backbone_out: None | torch.Tensor = None

        for name, backbone in zip(self._backbone_names, self._backbones):
            backbone = backbone.to(x.device)

            if name in ("rgb_i3d", "s3d"):
                feats = forward_2d_net(backbone, x.clone(), self._reduce)
            elif name in ("clip_vision"):
                continue
                # feats: Any = backbone.to(x.device)(x.clone().permute(0, 2, 1, 3, 4).reshape(-1, C, H, W))




        # print(backbone_out.shape)
        # neck_out = self._neck(backbone_out)
        # preds = self._head(neck_out)
        # return BaseModelOutput(
        #     preds,
        #     backbone_out if self._return_backbone_feats else None,
        #     neck_out if self._return_backbone_feats else None
        # )

    def _resolve_backbone_feat(self, feats: Any, B, C, T) -> torch.Tensor:
        """
        :param feats: extracted feature from backbone. Feats can be any datatype
        :return: tensor feats with shape (batch_size, hidden_dim)
        """
        if isinstance(feats, dict):
            # Model created by create_feature_extractor() form torchvision
            if "features" in feats.keys():
                feats: Any = feats["features"]

        # CLIP model output
        if isinstance(feats, BaseModelOutputWithPooling):
            feats: torch.Tensor = feats[1]

        print(feats.shape)
        # Reduce timestep to  3D-CNN-based model
        # if feats.dim() > 2:
        #     if self._reduce == "mean":
        #         feats = feats.mean(dim=list(range(2, feats.dim())))
        #     elif self._reduce == "first":
        #         feats = feats[..., 0]
        return feats


@dispatch(dict)
def resolve_net_output(x: Dict[str, Any]) -> Any:
    if "features" in x.keys():
        x = x["features"]
    return x


def forward_2d_net(model: torch.nn.Module, x: torch.Tensor, reduce: str) -> torch.Tensor:
    x: Any = model(x)
    x: torch.Tensor = resolve_net_output(x)

    # x = x.mean(dim=[2, 3, 4])
    print(x.shape)
    # if reduce == "mean":
    #     x = x.mean(dim=, keepdim=False)
    # elif reduce == "first":
    return x