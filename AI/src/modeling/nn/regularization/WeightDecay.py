from abc import abstractmethod

import torch
from torch.nn import Module
from torch.utils.hooks import RemovableHandle


__all__ = ["WeightDecay"]


class WeightDecay(Module):
    """
    Base class for weight decay regularization.
    """
    def __init__(self, module: Module, weight_decay: float, name: str = None):
        """
        :param module: pytorch Module
        :param weight_decay: Strength of regularization (has to be greater than `0.0`).
        :param name: Name of parameter to be regularized (if any).
            Default: all parameters will be regularized (including "bias").
        """
        if weight_decay <= 0.0:
            raise ValueError(
                "Regularization's weight_decay should be greater than 0.0, got {}".format(
                    weight_decay
                )
            )

        super(WeightDecay, self).__init__()
        self.module: Module = module
        self.weight_decay: float = weight_decay
        self.name: str = name
        self.hook: RemovableHandle = self.module.register_full_backward_hook(self._weight_decay_hook)

    def remove(self):
        self.hook.remove()

    def _weight_decay_hook(self, *_) -> None:
        if self.name is None:
            for param in self.module.parameters():
                if param.grad is None or torch.all(param.grad == torch.tensor(0.0)):
                    param.grad = self.regularize(param)
        else:
            for name, param in self.module.named_parameters():
                if self.name in name and (
                    param.grad is None or torch.all(param.grad == torch.tensor(0.0))
                ):
                    param.grad = self.regularize(param)

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def extra_repr(self) -> str:
        representation = "weight_decay={}".format(self.weight_decay)
        if self.name is not None:
            representation += ", name={}".format(self.name)
        return representation

    @abstractmethod
    def regularize(self, parameter):
        pass
