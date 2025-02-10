import dataclasses
from typing import Optional

import torch


__all__ = ["BaseModelOutput"]


@dataclasses.dataclass
class BaseModelOutput:
    score: torch.Tensor
    backbone_out: Optional[torch.Tensor] = None
    neck_out: Optional[torch.Tensor] = None
