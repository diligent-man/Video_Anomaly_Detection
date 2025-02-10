import copy
import torch


from ...utils import DotDict
from ..backbones import build_backbone
from ..necks import build_neck
# from ..heads import build_head

__all__ = ["BaseModel"]


class BaseModel(torch.nn.Module):
    def __init__(self, config: DotDict):
        super(BaseModel, self).__init__()
        # in_channels = config.get("in_channels", 3)
        # model_type = config["model_type"]

        # build transfrom,
        # if "Transform" not in config or config["Transform"] is None:
        #     self.use_transform = False
        # else:
        #     self.use_transform = True
        #     config["Transform"]["in_channels"] = in_channels
        #     self.transform = build_transform(config["Transform"])
        #     in_channels = self.transform.out_channels

        if config.Architecture.get("backbone") is None:
            self._backbone = None
        else:
            backbone, out_channel = build_backbone(copy.deepcopy(config))
            self._backbone = backbone

        # build neck
        if config.Architecture.get("neck") is None:
            self._neck = None
        else:
            self._neck = build_neck(copy.deepcopy(config))

        #     config["Neck"]["in_channels"] = in_channels
        #     self.neck = build_neck(config["Neck"])
        #     in_channels = self.neck.out_channels
        #
        # # # build head, head is need for det, rec and cls
        # if "Head" not in config or config["Head"] is None:
        #     self.use_head = False
        # else:
        #     self.use_head = True
        #     config["Head"]["in_channels"] = in_channels
        #     self.head = build_head(config["Head"])
        #
        # self.return_all_feats = config.get("return_all_feats", False)
