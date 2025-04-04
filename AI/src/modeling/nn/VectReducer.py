from typing import Dict, List
from functools import partial

import torch
from torch import Tensor
from torch.nn import Module
from torch.linalg import vector_norm


__all__ = ["VectReducer"]


class VectReducer(Module):
    __REDUCERS: Dict[str, partial] = {
        "max_l1": partial(vector_norm, ord=1, keepdim=True),
        "min_l1": partial(vector_norm, ord=1, keepdim=True),

        "max_l2": partial(vector_norm, ord=2, keepdim=True),
        "min_l2": partial(vector_norm, ord=2, keepdim=True),

        "max_inf": partial(vector_norm, ord=torch.inf, keepdim=True),
        "min_inf": partial(vector_norm, ord=torch.inf, keepdim=True),

        "max_-inf": partial(vector_norm, ord=-torch.inf, keepdim=True),
        "min_-inf": partial(vector_norm, ord=-torch.inf, keepdim=True)
    }

    def __init__(self,
                 cond: str,
                 ) -> None:
        super(VectReducer, self).__init__()
        assert cond in self.__REDUCERS.keys(), ValueError(
            f"Unavailable reduce condition. Currently support {self.__REDUCERS}. Get '{cond}'"
        )
        self.__cond: str = cond
        self.__reducer: partial = self.__REDUCERS[cond]

    def forward(self, x: Tensor, vec_dim: int, reduce_dim: int) -> Tensor:
        """
        :param x: input tensor with shape (B, ...)
        :param vec_dim: vector dim to perform reducer operation
        :param reduce_dim: dim to select vector based on reduced x
        :return: selected x with reduce dim by cond on vec dim
        """

        assert vec_dim < x.dim(), ValueError("Vector dimension is out of range")
        assert reduce_dim < x.dim(), ValueError("Reduce dimension is out of range")

        reduced_x: Tensor = self.__reducer(x=x, dim=vec_dim)

        if self.__cond.startswith("max"):
            reduced_x: List[int] = torch.argmax(reduced_x, reduce_dim).flatten().tolist()
        else:
            reduced_x: List[int] = torch.argmin(reduced_x, reduce_dim).flatten().tolist()

        shape: List[int] = list(x.shape)
        shape[reduce_dim] = 1

        out: Tensor = torch.zeros(shape, dtype=x.dtype, layout=x.layout, device=x.device)
        for i in range(len(reduced_x)):
            out[i, ...] = x[i, ...].select(reduce_dim-1, reduced_x[i]).unsqueeze(reduce_dim-1)
        return out
