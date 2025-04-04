from typing import Union, List, Dict

import torch
from torch import Tensor

__all__ = ["to_float32"]


def to_float32(inps: Union[Tensor, List[Tensor], Dict[str, Tensor]]) -> Union[Tensor, List[Tensor], Dict[str, Tensor]]:
    # cast to float32 in case of using torch.amp.autocast, cuz dtype != float32 can't precisely visualize
    if isinstance(inps, dict):
        for k in inps:
            if isinstance(inps[k], dict) or isinstance(inps[k], list):
                inps[k] = to_float32(inps[k])
            elif isinstance(inps[k], Tensor):
                inps[k] = inps[k].type(torch.float32)

    elif isinstance(inps, list):
        for k in range(len(inps)):
            if isinstance(inps[k], dict):
                inps[k] = to_float32(inps[k])
            elif isinstance(inps[k], list):
                inps[k] = to_float32(inps[k])
            elif isinstance(inps[k], Tensor):
                inps[k] = inps[k].type(torch.float32)

    elif isinstance(inps, Tensor):
        inps = inps.type(torch.float32)
    return inps



