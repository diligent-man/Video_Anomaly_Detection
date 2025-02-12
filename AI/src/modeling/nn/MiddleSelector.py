import torch

from .functional import middle_selector

__all__ = ["MiddleSelector"]


class MiddleSelector(torch.nn.Module):
    def __init__(self):
        super(MiddleSelector, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return middle_selector(x)