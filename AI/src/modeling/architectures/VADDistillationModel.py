import os.path
import warnings

from typing import Dict, Any, List
from collections import defaultdict


import torch

from torch import Tensor
from torch.nn import Module


from .BaseModel import BaseModel
from .BaseModelOutput import BaseModelOutput

from ...utils import DotDict, freeze_layer


__all__ = ["VADDistillationModel"]


class VADDistillationModel(Module):
    """
    Build model from the idea of offline distillation. Code logic is adopted from PaddleOCR
    """
    models: Dict[str, List[Module]] = defaultdict(list)

    def __init__(self, config: DotDict) -> None:
        super(VADDistillationModel, self).__init__()
        config = config.Architecture

        for model_type in config.models.get_dict():
            return_neck_out: bool = config.models[model_type].neck.get("return_neck_out", None)
            if return_neck_out is None or return_neck_out is False:
                config.models[model_type].neck["return_neck_out"] = True
                warnings.warn(f"return_neck_out was set to True instead of {return_neck_out} in config cuz offline "
                              f"distillation requires intermediate feat from teacher model"
                              )

            if model_type.startswith("student"):
                model: Module = BaseModel(config.models[model_type])
                self.models["student"].append(model)
            else:
                model: Module = BaseModel(config.models[model_type])
                trainable_layers: True = config.models[model_type].get("trainable_layers", 0)
                assert trainable_layers == 0, ValueError("All layers must be non-trainable in offline distillation training")

                pretrained: str = config.models[model_type].get("pretrained", None)
                assert pretrained is not None, ValueError(f"Weight for {model_type} must be provided in offline distillation training")
                assert os.path.isfile(pretrained), ValueError("Provided path is not a file")
                assert pretrained.endswith((".pt", ".pth")), ValueError("Provided file does not ends with '.pt' or '.pth'")

                ckpt: Dict[str, Any] = torch.load(pretrained)
                model.load_state_dict(ckpt["model"] if isinstance(ckpt["model"], dict) else ckpt["model"].state_dict())
                model, _, _ = freeze_layer(model, trainable_layers)
                self.models["teacher"].append(model)

                # model.train()
                # for name, para in model.named_parameters():
                #     print(name, para.requires_grad)

    def __repr__(self) -> str:
        prompt = (f"{self.__class__.__name__} includes {self.models['teacher'].__len__()} teacher(s) "
                  f"and {self.models['student'].__len__()} students")
        return prompt

    def forward(self, anomaly: Tensor, normal: Tensor, device="cpu") -> Dict[str, List[BaseModelOutput]]:
        """
        :param anomaly: list of input tensors in the format of Shape (S,C,T,H,W) or (B,S,C,T,H,W)
        :param normal:                                  //
        :return: BaseModelOutput obj
        """
        outs: Dict[str, List[BaseModelOutput]] = defaultdict(list)

        for model_type in ["student", "teacher"]:
            for student in self.models[model_type]:
                for inp in (anomaly, normal):
                    outs[model_type].append(student(inp.to(device)))  # currently placed on cuda
        return outs
