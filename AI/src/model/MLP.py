import copy
import collections

from typing import List, Tuple, Dict, Any


import torch


from ..opensrc.pytorch import available_layer
__all__ = ["MLP"]


class MLP(torch.nn.Module):
    """
    Multi-layer perceptron with alternate pattern as follows:
        a/ fc -> act -> drop
        hidden_layer  --> activation --> dropout --  + output_activation (if have)
              ↑----------------------------------|

        a/ fc -> drop -> act
        hidden_layer  --> dropout --> activation --  + output_activation (if have)
              ↑----------------------------------|
    """
    __hidden_layers: torch.nn.Sequential
    __out_activation: torch.nn.Module | None
    __layers: torch.nn.Sequential

    def __init__(self,
                 input_dim: int,
                 hidden_dim: List[int] | Tuple[int] | int,
                 output_dim: int,
                 bias: bool = True,
                 dropout: float = None,
                 hidden_activation: str = None,
                 hidden_activation_args: Dict[str, Any] = None,
                 out_activation: str = None,
                 out_activation_args: Dict[str, Any] = None,
                 layer_order: str = "fc->drop->act",
                 device: torch.device = None,
                 dtype: torch.dtype = None
                 ) -> None:
        super(MLP, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        if hidden_activation is not None:
            hidden_activation_args: Dict[str, Any] = dict() if hidden_activation_args is None else hidden_activation_args
            hidden_activation: torch.nn.Module = self._init_activation(hidden_activation, **hidden_activation_args)

        if out_activation is not None:
            out_activation_args: Dict[str, Any] = dict() if out_activation_args is None else out_activation_args
            out_activation: torch.nn.Module = self._init_activation(out_activation, **out_activation_args)

        dropout = torch.nn.Dropout(dropout) if dropout is not None and dropout > 0 else None

        self._layers = self.__make_layers(
            input_dim,
            hidden_dim,
            output_dim,
            bias,
            dropout,
            hidden_activation,
            out_activation,
            layer_order,
            **factory_kwargs
        )
    @property
    def layers(self) -> torch.nn.Sequential:
        return self.__layers

    @staticmethod
    def _init_activation(activation: str, **kwargs) -> torch.nn.Module:
        if activation is None or activation not in available_layer.keys():
            print(f"Apply default activation function: {torch.nn.ReLU.__name__}")
            return torch.nn.ReLU(**kwargs)
        else:
            return available_layer[activation](**kwargs)

    @staticmethod
    def __make_layers(input_dim: int,
                      hidden_dim: List[int] | Tuple[int] | int,
                      output_dim: int,
                      bias: bool,
                      dropout: torch.nn.Module | None,
                      hidden_activation: torch.nn.Module | None,
                      out_activation: torch.nn.Module | None,
                      layer_order: str,
                      **kwargs
                      ) -> torch.nn.Sequential:
        if isinstance(hidden_dim, int):
            hidden_dim: List[int] = [input_dim, hidden_dim, output_dim]
        else:
            hidden_dim: List[int] = [input_dim, *hidden_dim, output_dim]

        hidden_layers: Dict = {}

        for i in range(1, len(hidden_dim)):
            fc_layer = torch.nn.Linear(hidden_dim[i-1], hidden_dim[i], bias, **kwargs)

            if layer_order == "fc->drop->act":
                hidden_layers[f"fc{i}"] = fc_layer

                if dropout is not None and i < len(hidden_dim) - 1:
                    hidden_layers[f"dropout{i}"] = copy.deepcopy(dropout)

                if hidden_activation is not None and i < len(hidden_dim) - 1:
                    hidden_layers[f"act{i}"] = copy.deepcopy(hidden_activation)
            elif layer_order == "fc->act->drop":
                hidden_layers[f"fc{i}"] = fc_layer

                if hidden_activation is not None and i < len(hidden_dim) - 1:
                    hidden_layers[f"act{i}"] = copy.deepcopy(hidden_activation)

                if dropout is not None and i < len(hidden_dim) - 1:
                    hidden_layers[f"dropout{i}"] = copy.deepcopy(dropout)

        if out_activation is not None:
            hidden_layers[f"out_act"] = out_activation
        return torch.nn.Sequential(collections.OrderedDict(hidden_layers))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._layers(x)
        return x
