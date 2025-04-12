import dataclasses
from typing import Optional, List

from torch import Tensor
from transformers.modeling_outputs import ModelOutput


__all__ = [
    "ModelOutput",
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
    feats: Optional[Tensor] = None
