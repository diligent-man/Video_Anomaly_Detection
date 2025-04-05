from copy import deepcopy
from collections import OrderedDict
from typing import List, Tuple, Dict, Any

import torch
from torch import Tensor
from torch.nn import Module, Sequential, Dropout, ReLU

from ...opensrc.pytorch import avail_act
from .regularization import avail_regularizers


__all__ = ["MLP"]


class MLP(Module):
    """
    Multi-layer perceptron with alternate pattern as follows:
        a/ fc -> act -> drop
        hidden_layer  --> activation --> dropout --  + output_activation (if have)
              ↑----------------------------------|

        a/ fc -> drop -> act
        hidden_layer  --> dropout --> activation --  + output_activation (if have)
              ↑----------------------------------|
    """
    layers: Sequential

    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 hid_dim: List[int] | Tuple[int] | int,
                 hid_act: str = None,
                 hid_act_args: Dict[str, Any] = None,
                 out_act: str = None,
                 out_act_args: Dict[str, Any] = None,
                 regularize: str = None,
                 regularize_args: Dict[str, Any] = None,
                 bias: bool = True,
                 dropout: float = None,
                 layer_order: str = "fc->drop->act",
                 device: torch.device = None,
                 dtype: torch.dtype = None
                 ) -> None:
        super(MLP, self).__init__()
        factory_kwargs = {"device": device, "dtype": dtype}

        for act in (hid_act, out_act):
            if act is not None:
                assert act in avail_act.keys(), ValueError("Provided activation is currently unavailable")

        if hid_dim is not None:
            hid_act_args: Dict[str, Any] = dict() if hid_act_args is None else hid_act_args
            hid_act: Module = self._init_act(hid_act, **hid_act_args)

        if out_act is not None:
            out_act_args: Dict[str, Any] = dict() if out_act_args is None else out_act_args
            out_act: Module = self._init_act(out_act, **out_act_args)

        dropout = Dropout(dropout) if dropout is not None and dropout > 0 else None

        self._in_channels: int = in_channels
        self._out_channels: int = out_channels
        self.layers = self.__make_layers(
            self._in_channels,
            hid_dim,
            self._out_channels,
            bias,
            dropout,
            hid_act,
            out_act,
            layer_order,
            **factory_kwargs
        )

        if regularize is not None:
            assert regularize in avail_regularizers.keys(), ValueError("Provided regularizer is currently unavailable")
            self.layers: Module = avail_regularizers[regularize](self.layers, **regularize_args)

    @property
    def in_channels(self) -> int:
        return self._in_channels

    @property
    def out_channels(self) -> int:
        return self._out_channels

    @staticmethod
    def _init_act(activation: str, **kwargs) -> Module:
        if activation is None or activation not in avail_act.keys():
            print(f"Apply default activation function: {ReLU.__name__}")
            return ReLU(**kwargs)
        else:
            return avail_act[activation](**kwargs)

    @staticmethod
    def __make_layers(input_dim: int,
                      hidden_dim: List[int] | Tuple[int] | int,
                      output_dim: int,
                      bias: bool,
                      dropout: Module | None,
                      hidden_activation: Module | None,
                      out_activation: Module | None,
                      layer_order: str,
                      **kwargs
                      ) -> Sequential:
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
                    hidden_layers[f"dropout{i}"] = deepcopy(dropout)

                if hidden_activation is not None and i < len(hidden_dim) - 1:
                    hidden_layers[f"act{i}"] = deepcopy(hidden_activation)
            elif layer_order == "fc->act->drop":
                hidden_layers[f"fc{i}"] = fc_layer

                if hidden_activation is not None and i < len(hidden_dim) - 1:
                    hidden_layers[f"act{i}"] = deepcopy(hidden_activation)

                if dropout is not None and i < len(hidden_dim) - 1:
                    hidden_layers[f"dropout{i}"] = deepcopy(dropout)

        if out_activation is not None:
            hidden_layers[f"out_act"] = out_activation

        hidden_layers: Sequential = Sequential(OrderedDict(hidden_layers))
        return hidden_layers

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        return x
