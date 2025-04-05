import gc
import os.path
import warnings
from typing import Dict, Any, List
from collections import defaultdict

import torch
from torch import Tensor
from torch.nn import Module

from ..nn import MLP
from ...utils import DotDict, freeze_layer
from . import BaseModel, BaseModelOutput, VADDistillModelOutput


__all__ = ["VADDistillationModel"]


class VADDistillationModel(Module):
    """
    Build model from the idea of offline distillation. Code logic is adopted from PaddleOCR
    """

    __FEAT_PREPROCESSING: Dict[str, Module] = {
        "MLP": MLP
    }

    def __init__(self, config: DotDict) -> None:
        super(VADDistillationModel, self).__init__()

        self.__config = config.Architecture
        self.__soft_label_threshold = self.__config.pop("soft_label_threshold", 0.5)

        self.feat_preprocessing: None | Module = self._build_feat_preprocessing()
        self.models: Dict[str, List[Module]] = self._build_student_teacher()

    def __repr__(self) -> str:
        prompt = (f"{self.__class__.__name__} includes {self.models['teacher'].__len__()} teacher(s) "
                  f"and {self.models['student'].__len__()} students")
        return prompt

    def _build_feat_preprocessing(self) -> None | Module:
        feat_preprocessing: None | DotDict = self.__config.pop("feat_preprocessing", None)
        if feat_preprocessing is not None:
            name: None | str = feat_preprocessing.pop("name", None)
            assert name in self.__FEAT_PREPROCESSING.keys(), ValueError(
                f"Specified feat preprocessing {name} is unavailable. "
                f"Currently support {self.__FEAT_PREPROCESSING.keys()}."
            )

            feat_preprocessing: Module = self.__FEAT_PREPROCESSING[name](**feat_preprocessing.get_dict())
        return feat_preprocessing

    def _build_student_teacher(self) -> Dict[str, List[Module]]:
        models: Dict[str, List[Module]] = defaultdict(list)

        for model_type in self.__config.models.get_dict():
            return_neck_out: bool = self.__config.models[model_type].neck.get("return_neck_out", None)
            if return_neck_out is None or return_neck_out is False:
                self.__config.models[model_type].neck["return_neck_out"] = True
                warnings.warn(f"return_neck_out was set to True instead of {return_neck_out} in config cuz offline "
                              f"distillation requires intermediate feat from teacher model"
                              )

            if model_type.startswith("student"):
                model: Module = BaseModel(self.__config.models[model_type])
                models["student"].append(model)
            else:
                model: Module = BaseModel(self.__config.models[model_type])
                trainable_layers: True = self.__config.models[model_type].get("trainable_layers", 0)
                assert trainable_layers == 0, ValueError("All layers must be non-trainable in offline distillation training")

                pretrained: str = self.__config.models[model_type].get("pretrained", None)
                assert pretrained is not None, ValueError(f"Weight for {model_type} must be provided in offline distillation training")
                assert os.path.isfile(pretrained), ValueError("Provided path is not a file")
                assert pretrained.endswith((".pt", ".pth")), ValueError("Provided file does not ends with '.pt' or '.pth'")

                ckpt: Dict[str, Any] = torch.load(pretrained)
                model.load_state_dict(ckpt["model"] if isinstance(ckpt["model"], dict) else ckpt["model"].state_dict())
                model, _, _ = freeze_layer(model, trainable_layers)
                models["teacher"].append(model)
        return models

    def forward(self, anomaly: Tensor, normal: Tensor, device="cpu") -> Dict[str, List[BaseModelOutput]]:
        """
        :param anomaly: list of input tensors in the format of Shape (S,C,T,H,W) or (B,S,C,T,H,W)
        :param normal:                                  //
        :param device: computing device
        :return: BaseModelOutput obj
        """
        outs: Dict[str, List[VADDistillModelOutput]] = defaultdict(list)

        for model_type in ["student", "teacher"]:
            for model in self.models[model_type]:
                for inp in (anomaly, normal):
                    model_out: BaseModelOutput = model(inp.to(device))

                    # Soft labels
                    soft_labels: Tensor = model_out.preds

                    # Hard labels
                    hard_labels: Tensor = torch.nn.Sigmoid()(model_out.preds)
                    hard_labels = torch.where(hard_labels >= self.__soft_label_threshold, 1., 0.)

                    feats: Tensor = model_out.neck_outs
                    if self.feat_preprocessing is not None:
                        feats = self.feat_preprocessing.to(device)(feats)


                    print(feats.shape)
                    exit()

                    # model_out.preds = torch.where(model_out.preds >= self.__soft_label_threshold, 1., 0.)
                    outs[model_type].append(model_out)  # currently placed on cuda
        self.feat_preprocessing.to("cpu")
        gc.collect()
        torch.cuda.empty_cache()
        return outs
