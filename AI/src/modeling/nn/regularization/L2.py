from torch import Tensor
from torch.nn import Parameter
from .WeightDecay import WeightDecay

__all__ = ["L2"]


class L2(WeightDecay):
    """
    Regularize module's parameters using L2 weight decay.
    """
    def regularize(self, para: Parameter) -> Tensor:
        return self.weight_decay * para.data
