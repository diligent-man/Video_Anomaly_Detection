from typing import Union, Iterable, Dict, List

import torch
from torch import Tensor
from torch.nn import Module


from .CombinedLoss import CombinedLoss
from .MILRankingLoss import MILRankingLoss


from ..opensrc.pytorch import avail_loss
from ..utils import DotDict, make_border, to_float32
from ..modeling.architectures import ModelOutput


__all__ = ["LossWrapper"]


class LossWrapper:
    __name: str
    __has_aux: bool
    __loss: Module
    __LOSSES: Dict[str, Module] = {
        **avail_loss,
        "MILRankingLoss": MILRankingLoss,
        "CombinedLoss": CombinedLoss
    }

    def __init__(self, config: DotDict) -> None:
        name: None | str = config.Loss.pop("name", None)
        has_aux: bool = config.Loss.pop("has_aux", False)
        assert name in self.__LOSSES.keys(), ValueError(f"Provided loss is invalid. Get '{name}'")

        self.__name = name
        self.__has_aux = has_aux
        self.__loss = self.__LOSSES[name](**config.Loss.get_dict())

        top, bottom = make_border("Build Loss")
        print(top)
        print(f"""Loss: {name}
Has aux logits: {self.__has_aux}""")
        print(bottom)

    @property
    def name(self) -> str:
        return self.__name

    def compute_batch_loss(self,
                           inputs: Union[Tensor, Iterable[Tensor], List[ModelOutput]],
                           targets: Union[Tensor, Iterable[Tensor], List[ModelOutput]] = None,
                           aux_logits_weight: float = 0.3,
                           ) -> torch.Tensor:
        is_list: bool = isinstance(inputs, list)
        if is_list is not True and isinstance(inputs[0], Tensor):
            inputs = [inputs]

        if targets is not None and is_list is not True:
            if isinstance(targets, Tensor):
                targets = [targets]

        inputs: List[Union[Tensor, ModelOutput]] = to_float32(inputs)
        is_tensor: bool = isinstance(inputs[0], Tensor)
        if self.__has_aux:
            # aux logits (GoogleLeNet, InceptionV3)
            # Not tested later
            batch_loss = [
                self.__loss(inputs[i], targets[i] if targets is not None else (None,)) for i in range(len(inputs))
            ]
            batch_loss = batch_loss[0] + sum(batch_loss) * aux_logits_weight
        else:
            if is_tensor:
                batch_loss = self.__loss(
                    *inputs, *targets if targets is not None else (None,)
                ) if targets is not None else self.__loss(*inputs)
            else:
                batch_loss = self.__loss(
                    inputs, targets if targets is not None else (None,)
                ) if targets is not None else self.__loss(inputs)
        return batch_loss
