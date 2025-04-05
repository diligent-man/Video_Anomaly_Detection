import dataclasses
from typing import Optional, List

import torch
from transformers.modeling_outputs import ModelOutput


__all__ = [
    "SimpleClassifierOutput"
]


@dataclasses.dataclass
class SimpleClassifierOutput(ModelOutput):
    preds: torch.Tensor = None
    logits: torch.Tensor = None
