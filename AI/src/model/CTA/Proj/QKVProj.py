from typing import Tuple

import torch

__all__ = ["QKVProj"]


class QKVProj(torch.nn.Module):
    def __init__(self,
                 embed_dim: int,
                 bias: bool = True,
                 device: torch.device = None,
                 dtype: torch.dtype = None,
                 ) -> None:
        super(QKVProj, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self._embed_dim: int = embed_dim

        self._in_proj_weight = torch.nn.Parameter(torch.empty((3 * embed_dim, embed_dim), **factory_kwargs))
        self._in_proj_bias = torch.nn.Parameter(torch.empty(3 * embed_dim, **factory_kwargs)) \
            if bias else self.register_parameter("_in_proj_bias", None)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        torch.nn.init.xavier_uniform_(self._in_proj_weight)

        if self._in_proj_bias is not None:
            torch.nn.init.constant_(self._in_proj_bias, 0)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        :param x: shape (B, Seq_len, embed_dim)
        :return:
        """
        batch_size, seq_len, embed_dim = x.size()
        x = x @ self._in_proj_weight.T

        if self._in_proj_bias is not None:
            x += self._in_proj_bias

        x = x.reshape((3, batch_size, seq_len, embed_dim))
        q, k, v = x[:, ...]
        return q, k, v
