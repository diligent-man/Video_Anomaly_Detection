from typing import Union, Sequence, Any, Mapping, Iterable

import torch
import torchinfo
import numpy as np


__all__ = ["ModelArchInspector"]


# Type aliases
INPUT_DATA_TYPE = Union[torch.Tensor, np.ndarray, Sequence[Any], Mapping[str, Any]]
INPUT_SIZE_TYPE = Sequence[Union[int, Sequence[Any], torch.Size]]


class ModelArchInspector(object):
    """
    See more at torchinfo.summary()
    """
    __model: torch.nn.Module
    __input_size: INPUT_SIZE_TYPE | None
    __input_data: INPUT_DATA_TYPE | None
    __col_names: Iterable[str]
    __col_width: int
    __depth: int
    __device: torch.device | str | None
    __dtypes: list[torch.dtype] | None
    __mode: str | None
    __verbose: int | None
    __col_names: Iterable[str]

    def __init__(self,
                 model: torch.nn.Module,
                 input_size: INPUT_SIZE_TYPE | None = None,
                 input_data: INPUT_DATA_TYPE | None = None,
                 col_names: Iterable[str] | None = ("input_size", "output_size", "num_params", "params_percent", "trainable"),
                 col_width: int = 25,
                 depth: int = 3,
                 device: torch.device | str | None = None,
                 dtypes: list[torch.dtype] | None = None,
                 mode: str | None = None,
                 verbose: int | None = 1,
                 **kwargs: Any
                 ):
        self.__model = model
        self.__input_size = input_size
        self.__input_data = input_data
        self.__col_names = col_names
        self.__col_width = col_width
        self.__depth = depth
        self.__device = device
        self.__dtypes = dtypes
        self.__mode = mode
        self.__verbose = verbose
        self.__kwargs = kwargs

    def __call__(self):
        torchinfo.summary(
            self.__model,
            self.__input_size,
            self.__input_data,
            None,
            None,
            self.__col_names,
            self.__col_width,
            self.__depth,
            self.__device,
            self.__dtypes,
            self.__mode,
            None,
            self.__verbose,
            **self.__kwargs
        )
