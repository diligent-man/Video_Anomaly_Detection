import dataclasses
from typing import Optional, List

import torch
from transformers.modeling_outputs import ModelOutput


__all__ = ["BaseModelOutput"]


@dataclasses.dataclass
class BaseModelOutput(ModelOutput):
    preds: torch.Tensor
    extracted_feats: Optional[List[torch.Tensor]] = None
    projected_feats: Optional[torch.Tensor] = None
    neck_outs: Optional[torch.Tensor] = None
