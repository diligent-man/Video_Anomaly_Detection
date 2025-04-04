from .WeightDecay import WeightDecay

import torch
from torch import Tensor
from torch.nn import Parameter


__all__ = ["L1"]


class L1(WeightDecay):
    """
    Regularize module's parameters using L1 weight decay.
    """
    def regularize(self, para: Parameter) -> Tensor:
        return self.weight_decay * torch.sign(para.data)
