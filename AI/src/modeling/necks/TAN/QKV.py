from typing import Tuple

import torch
from torch import Tensor
from torch.nn import Parameter


__all__ = ["QKV"]


class QKV(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 bias: bool = True,
                 device: torch.device = None,
                 dtype: torch.dtype = None,
                 ) -> None:
        super(QKV, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self._embed_dim: int = embed_dim
        self.weight = Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self.bias = Parameter(torch.empty(3 * embed_dim, **factory_kwargs)) \
            if bias else self.register_parameter("bias", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self.weight)

        if self.bias is not None:
            torch.nn.init.constant_(self.bias, 0)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        :param x: Shape (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
        :return:
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x

        x = x @ self.weight.T

        if self.bias is not None:
            x += self.bias

        q, k, v = x.chunk(3, dim=-1)
        return q, k, v
