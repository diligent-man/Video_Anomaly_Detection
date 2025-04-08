from typing import List, Optional

import torch
from torch import Tensor
from torch.nn import BCEWithLogitsLoss

from .DistillationLoss import DistillationLoss
from ...modeling.architectures import VADDistillModelOutput


__all__ = ["DistillBCEWithLogitsLoss"]


class DistillBCEWithLogitsLoss(DistillationLoss):
    """
    Treat every pair of models as a single loss, did not implement the concept called "ensemble learning"
    Due to equivalent treatment, final loss is a mean of all pairs
    """
    def __init__(self,
                 key: str,
                 model_idx_pairs: List[List[int]],
                 weight: Optional[Tensor] = None,
                 reduction: str = "mean",
                 pos_weight: Optional[Tensor] = None,
                 name="BCELoss",
                 ) -> None:
        assert reduction in ["none", "mean", "sum", None], ValueError
        super(DistillBCEWithLogitsLoss, self).__init__(key, model_idx_pairs)
        self.__name = name

        self.__reduction: str = reduction
        self.__loss = BCEWithLogitsLoss(weight, reduction="mean", pos_weight=pos_weight)

    def forward(self, student_outs: List[VADDistillModelOutput], teacher_outs: List[VADDistillModelOutput]) -> Tensor:
        """
        :param student_outs: VADDistillModelOutput that contains soft_preds (B, S)
        :param teacher_outs:                        //
        :return: computed loss
        """
        loss: None | Tensor = None
        for i, pair_idx in enumerate(self._model_idx_pairs):
            result = self.__loss(student_outs[pair_idx[0]][self._key], teacher_outs[pair_idx[1]][self._key])
            loss = result if loss is None else torch.vstack((loss, result))

        if self.__reduction == "mean":
            loss: Tensor = torch.mean(loss, 0)
        elif self.__reduction == "sum":
            loss: Tensor = torch.sum(loss, 0)
        return loss
