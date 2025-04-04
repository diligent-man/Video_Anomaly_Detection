from typing import List, Dict, Any

import torch
from torch import Tensor
from torch.nn import Module

from .distillation import avail_dl_loss
from ..modeling.architectures import BaseModelOutput


__all__ = ["CombinedLoss"]


class CombinedLoss(Module):
    """
    A combination of loss function with respective weight
    """
    __loss_lst: List[Module] = []
    __combined_weight_lst: List[int | float] = []

    def __init__(self, loss_config_lst: List[Dict[str, Any]]) -> None:
        super(CombinedLoss, self).__init__()
        assert isinstance(loss_config_lst, list), ValueError(f"Get {type(loss_config_lst)} instead of list of dict")

        for config in loss_config_lst:
            assert isinstance(config, dict), ValueError(f"Get {type(config)} instead of dict")

            name: None | str = config.pop("name", None)
            combined_weight: None | float = config.pop("combined_weight", None)

            assert name in avail_dl_loss.keys(), ValueError(f"{name} is not supported in combined loss")
            assert combined_weight is not None, ValueError(f"Get {type(combined_weight)} instead of float")

            loss: Module = avail_dl_loss[name](**config)

            self.__loss_lst.append(loss)
            self.__combined_weight_lst.append(combined_weight)

    def forward(self, student_outs: List[BaseModelOutput], teacher_outs: List[BaseModelOutput]):
        return_loss: None | Tensor = None

        for i, loss_fn in enumerate(self.__loss_lst):
            loss: Tensor = loss_fn(student_outs, teacher_outs)
            loss *= self.__combined_weight_lst[i]

            if return_loss is None:
                return_loss = loss
            else:
                return_loss = torch.vstack((return_loss, loss))

        return_loss = return_loss.sum(dim=0)
        return return_loss
