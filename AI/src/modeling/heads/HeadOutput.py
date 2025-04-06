import dataclasses

from torch import Tensor
from transformers.modeling_outputs import ModelOutput


__all__ = [
    "SimpleClassifierOutput"
]


@dataclasses.dataclass
class SimpleClassifierOutput(ModelOutput):
    preds: Tensor = None
    logits: Tensor = None
