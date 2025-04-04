from functools import partial
from typing import Optional, Union, Tuple

import torch
from torch import Tensor

from torch.nn import Module


__all__ = ["VectorNorm"]


class VectNorm(Module):
    """
    Class that encompasses torch.linalg.vector_norm() fn.
    """
    def __init__(self,
                 dim: Optional[Union[int, Tuple[int]]],
                 ord: Optional[Union[int, float]],
                 keepdim: Optional[bool] = False
                 ) -> None:
        super().__init__()
        self.vector_norm: partial = partial(torch.linalg.vector_norm, dim=dim, ord=ord, keepdim=keepdim)

    def forward(self, x: Tensor) -> Tensor:
        print(torch.linalg.vector_norm(x, dim=2, keepdim=True)[0, ...])
        x = self.vector_norm(x)

        print(x[0, ...])
        return x
