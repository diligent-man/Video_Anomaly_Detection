import collections

from typing import List, Tuple, Dict

import torch


__all__ = ["MLP"]


class MLP(torch.nn.Module):
    """
    Multi-layer perceptron with alternate pattern as follows:
        hidden_layer  --> activation --> dropout --  + output_activation (if have)
              ↑----------------------------------|
    """
    __hidden_layers: torch.nn.Sequential
    __out_activation: torch.nn.Module | None

    def __init__(self,
                 input_dim: int,
                 hidden_dim: List[int] | Tuple[int] | int,
                 output_dim: int,
                 bias: bool = True,
                 dropout: float = None,
                 hidden_activation: torch.nn.Module = None,
                 out_activation: torch.nn.Module = None,
                 device: torch.device = None,
                 dtype: torch.dtype = None
                 ) -> None:
        super(MLP, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        self.__hidden_layers: torch.nn.Sequential = self.__make_layers(
            input_dim,
            hidden_dim,
            output_dim,
            bias,
            dropout,
            hidden_activation,
            **factory_kwargs
        )

        self.__out_activation = out_activation

    @staticmethod
    def __make_layers(input_dim: int,
                      hidden_dim: List[int] | Tuple[int] | int,
                      output_dim: int,
                      bias: bool,
                      dropout: float | None,
                      hidden_activation: torch.nn.Module | None,
                      **kwargs
                      ) -> torch.nn.Sequential:
        if isinstance(hidden_dim, int):
            hidden_dim: List[int] = [input_dim, hidden_dim, output_dim]
        else:
            hidden_dim: List[int] = [input_dim, *hidden_dim, output_dim]

        hidden_layers: Dict = {}

        for i in range(1, len(hidden_dim)):
            hidden_layers[f"fc_{i}"] = torch.nn.Linear(hidden_dim[i-1], hidden_dim[i], bias, **kwargs)

            if hidden_activation is not None and i < len(hidden_dim)-1:
                hidden_layers[f"act_{i}"] = hidden_activation

            if dropout is not None and i < len(hidden_dim)-1:
                hidden_layers[f"dropout_{i}"] = torch.nn.Dropout(dropout)
        return torch.nn.Sequential(collections.OrderedDict(hidden_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__hidden_layers(x)

        if self.__out_activation is not None:
            x = self.__out_activation(x)
        return x
