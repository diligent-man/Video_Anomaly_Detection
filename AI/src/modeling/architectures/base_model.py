import copy
import torch


from ...utils import DotDict
from ..backbones import build_backbone
# from ..necks import build_neck
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
        if config.Architecture.backbone is None:
            self.use_backbone = False
        else:
            self.use_backbone = True
            # config.Backbone["in_channels"] = in_channels
            self.backbone = build_backbone(copy.deepcopy(config))
            # in_channels = self.backbone.out_channels

        # build neck
        # for rec, neck can be cnn,rnn or reshape(None)
        # for det, neck can be FPN, BIFPN and so on.
        # for cls, neck should be none
        # if "Neck" not in config or config["Neck"] is None:
        #     self.use_neck = False
        # else:
        #     self.use_neck = True
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
