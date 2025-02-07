from typing import Tuple

import torch

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

        self._w = torch.nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self._b = torch.nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs)) \
            if bias else self.register_parameter("_b", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self._w)

        if self._b is not None:
            torch.nn.init.constant_(self._b, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: Shape (seq_len, hidden_dim) or (batch, seq_len, hidden_dim)
        :return:
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x

        x = x @ self._w.T

        if self._b is not None:
            x += self._b

        q, k, v = x.chunk(3, dim=-1)
        return q, k, v
