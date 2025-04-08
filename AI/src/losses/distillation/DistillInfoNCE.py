from typing import List, Tuple

import torch
from torch import Tensor

from ..InfoNCE import InfoNCE
from .DistillationLoss import DistillationLoss
from ...modeling.architectures import VADDistillModelOutput


__all__ = ["DistillInfoNCE"]


class DistillInfoNCE(DistillationLoss):
    """
    Ref: Distilling Aggregated Knowledge for Weakly-Supervised Video Anomaly Detection
    """
    def __init__(self,
                 pred_key: str,
                 feat_key: str,
                 model_idx_pairs: List[List[int]],
                 reduction: str = "mean",
                 temperature: float = .1,
                 negative_mode: str = "unpaired",
                 name: str = "InfoNCELoss"
                 ) -> None:
        assert reduction in ["none", "mean", "sum", None], ValueError
        super(DistillInfoNCE, self).__init__([pred_key, feat_key], model_idx_pairs)

        self.__name: str = name
        self.__reduction: str = reduction
        self.__loss: InfoNCE = InfoNCE(temperature, "mean", negative_mode)

    def _prep_loss_inp(self, batch_idx: int, feat_type: int,
                       pos_feat_idx: Tensor, neg_feat_idx: Tensor,
                       teacher_feats: Tensor, student_feats: Tensor
                       ) -> Tuple[Tensor, Tensor, Tensor]:
        # (N, 2)
        pos_indices: Tensor = pos_feat_idx[pos_feat_idx[:, 0] == batch_idx]
        neg_indices: Tensor = neg_feat_idx[neg_feat_idx[:, 0] == batch_idx]

        # (N, Hid_dim)
        teacher_pos_feats: Tensor = teacher_feats[batch_idx, pos_indices[:, 1] if feat_type == 1 else neg_indices[:, 1], ...]
        student_pos_feats: Tensor = student_feats[batch_idx, pos_indices[:, 1] if feat_type == 1 else neg_indices[:, 1], ...]

        # (M, Hid_dim)
        student_neg_feats: Tensor = student_feats[batch_idx, neg_indices[:, 1] if feat_type == 1 else pos_indices[:, 1], ...]

        if self.__loss.negative_mode == "paired":
            # (N, M, Hid_dim)
            student_neg_feats = torch.repeat_interleave(student_neg_feats.unsqueeze(0), teacher_pos_feats.shape[0], dim=0)
        return teacher_pos_feats, student_pos_feats, student_neg_feats

    def forward(self, student_outs: List[VADDistillModelOutput], teacher_outs: List[VADDistillModelOutput]) -> Tensor:
        """
        Respectively compute
            anomaly part
                pull: (teacher_anomaly, student anomaly)
                push: (teacher_anomaly, student normal+teacher_anomaly)

            normal part
                pull: (teacher_normal, student_normal)
                push: (teacher_normal, student anomaly + teacher_normal)

            loss = (anomaly part + normal part) * temperature ** 2
        :param student_outs: VADDistillModelOutput that contains
            feats: shape (B, S, Hid_dim)
        :param teacher_outs: VADDistillModelOutput that contains
            hard_preds: shape (B, S)
            feats: shape (B, S, Hid_dim)
        :return: computed loss value
        """
        loss: None | Tensor = None
        for i, pair_idx in enumerate(self._model_idx_pairs):
            pos_feat_idx = torch.argwhere(teacher_outs[pair_idx[1]][self._key[0]] == 1)  # (M, 2)
            neg_feat_idx = torch.argwhere(teacher_outs[pair_idx[1]][self._key[0]] == 0)  # (M, 2)

            for batch_idx in range(pos_feat_idx.max(0).values[0]+1):
                batch_loss: None | Tensor = None

                for feat_type in (1, 0):
                    (
                        teacher_pos_feats,
                        student_pos_feats,
                        student_neg_feats
                    ) = self._prep_loss_inp(batch_idx, feat_type,
                                            pos_feat_idx, neg_feat_idx,
                                            student_outs[pair_idx[0]][self._key[1]].clone(),
                                            teacher_outs[pair_idx[1]][self._key[1]].clone()
                                            )
                    return_loss = self.__loss(teacher_pos_feats, student_pos_feats, student_neg_feats)
                    batch_loss = return_loss if batch_loss is None else batch_loss + return_loss

                loss = batch_loss if loss is None else torch.vstack((loss, batch_loss))
        if self.__reduction == "mean":
            loss: Tensor = torch.mean(loss, 0)
        elif self.__reduction == "sum":
            loss: Tensor = torch.sum(loss, 0)
        return loss * self.__loss.temperature ** 2
