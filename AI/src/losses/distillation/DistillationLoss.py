from typing import List, Dict

import torch
from torch import Tensor
from torch.nn import Module


class DistillationLoss(Module):
    _loss_dict: Dict[str, Tensor] = {}

    def __init__(self,
                 key: str,
                 model_idx_pairs: List[List[int]]
                 ) -> None:
        super(DistillationLoss, self).__init__()
        self._key: str = key
        self._model_idx_pairs: List[List[int]] = model_idx_pairs
