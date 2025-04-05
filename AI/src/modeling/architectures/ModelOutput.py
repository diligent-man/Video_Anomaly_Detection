import dataclasses
from typing import Optional, List

from torch import Tensor
from transformers.modeling_outputs import ModelOutput


__all__ = [
    "BaseModelOutput",
    "VADDistillModelOutput"
]


@dataclasses.dataclass
class BaseModelOutput(ModelOutput):
    extracted_feats: Optional[List[Tensor]] = None
    projected_feats: Optional[Tensor] = None
    neck_outs: Optional[Tensor] = None
    logits: Optional[Tensor] = None
    preds: Tensor = None


@dataclasses.dataclass
class VADDistillModelOutput(ModelOutput):
    soft_preds: Tensor = None
    hard_preds: Tensor = None
    teacher_pos_feats: Optional[Tensor] = None
    teacher_neg_feats: Optional[Tensor] = None
    student_pos_feats: Optional[Tensor] = None
    student_neg_feats: Optional[Tensor] = None
