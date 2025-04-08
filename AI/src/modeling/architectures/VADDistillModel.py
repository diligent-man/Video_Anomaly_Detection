import os
import warnings
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Callable

import torch
from torch import Tensor
from torch.nn import Module

from ..nn import MLP
from ...utils import DotDict, freeze_layer
from . import BaseModel, BaseModelOutput, VADDistillModelOutput


__all__ = ["VADDistillModel"]


class VADDistillModel(Module):
    """
    Build model from the idea of offline distillation. Code logic is adopted from PaddleOCR
    """

    __FEAT_PREPROCESSING: Dict[str, Module] = {
        "MLP": MLP
    }

    def __init__(self, config: DotDict) -> None:
        super(VADDistillModel, self).__init__()

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
        """
        :return: Build student and teacher with specified config in respective manner
        """
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

    def _forward_student(self, anomaly: Tensor, normal: Tensor, model: Module) -> VADDistillModelOutput:
        # Outs includes: logits & feats
        anomaly_outs: BaseModelOutput = model(anomaly)
        normal_outs: BaseModelOutput = model(normal)

        soft_preds: Tensor = torch.cat((anomaly_outs.logits, normal_outs.logits), dim=1)  # (B,2*S)

        feats: Tensor = torch.cat((anomaly_outs.neck_outs, normal_outs.neck_outs), dim=1)  # (B,2*S,Hid_dim)
        feats = self.feat_preprocessing.to(feats.device)(feats)
        return VADDistillModelOutput(soft_preds=soft_preds, feats=feats)

    def _forward_teacher(self, anomaly: Tensor, normal: Tensor, model: Module) -> VADDistillModelOutput:
        # Outs includes: feats, logits, preds
        anomaly_outs: BaseModelOutput = model(anomaly)
        normal_outs: BaseModelOutput = model(normal)

        feats: Tensor = torch.cat((anomaly_outs.neck_outs, normal_outs.neck_outs), dim=1)  # (B,2*S,Hid_dim)
        feats = self.feat_preprocessing.to(feats.device)(feats)

        hard_preds: Tensor = torch.cat((anomaly_outs.preds, normal_outs.preds), dim=1)
        hard_preds = torch.where(hard_preds >= self.__soft_label_threshold, 1., 0.)  # (B,2*S)

        soft_preds: Tensor = torch.cat((anomaly_outs.logits, normal_outs.logits), dim=1)  # (B,2*S)
        return VADDistillModelOutput(soft_preds=soft_preds, hard_preds=hard_preds, feats=feats)

    def forward(self, anomaly: Tensor, normal: Tensor, device: str)\
            -> Tuple[List[VADDistillModelOutput], List[VADDistillModelOutput]]:
        """
        :param anomaly: list of input tensors in the format of Shape (S,C,T,H,W) or (B,S,C,T,H,W)
        :param normal:                                  //
        :param device: computing device
        :return: BaseModelOutput obj

        Student's out:
            neck_outs: shape (B,S,Hid_dim)
            logits (soft labels): shape (B,S)

        Teacher's out:
            neck_outs: shape (B,S,Hid_dim)
            logits (soft labels): shape (B,S)
            prob (hard labels): shape (B,S)
        """
        outs: Dict[str, List[VADDistillModelOutput]] = defaultdict(list)

        for model_type in ["student", "teacher"]:
            for model in self.models[model_type]:
                fn: Callable = getattr(self, f"_forward_{model_type}")
                outs[model_type].append(fn(anomaly.to(device), normal.to(device), model))
        return outs["student"], outs["teacher"]
