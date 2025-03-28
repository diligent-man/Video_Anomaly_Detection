import torch

from torch import Tensor
from torch.nn import Module

__all__ = ["MILRankingLoss"]


class MILRankingLoss(Module):
    """
    Ref: Real-world Anomaly Detection in Surveillance Videos
    https://github.com/seominseok0429/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch/blob/main/loss.py
    """
    def __init__(self,
                 topk: int = 1,
                 margin: float = 0.0,
                 sparsity_constraint: bool = False,
                 smoothness_constraint: bool = False,
                 sparsity_weight: float = 8 * 1e-5,
                 smoothness_weight: float = 8 * 1e-5,
                 reduction: str = "mean",
                 ) -> None:
        super(MILRankingLoss, self).__init__()
        assert topk == 1, ValueError("Currently support top 1")
        self.__topk: int = topk
        self.__margin: float = margin
        self.__sparsity_constraint: bool = sparsity_constraint
        self.__smoothness_constraint: bool = smoothness_constraint
        self.__sparsity_weight: float = sparsity_weight
        self.__smoothness_weight: float = smoothness_weight
        self.__reduction: str = reduction

    def forward(self, anomaly_preds: Tensor, normal_preds: Tensor) -> Tensor:
        """
        :param normal_preds: (B, S)
        :param anomaly_preds: (B, S)
        :return: torch.float scalar
        """
        # Currently run with topk = 1
        top_k_normal_preds: Tensor = normal_preds.max(dim=1).values
        top_k_anomaly_preds: Tensor = anomaly_preds.max(dim=1).values

        loss: Tensor = self.__margin - 1 * (top_k_anomaly_preds - top_k_normal_preds)
        loss = torch.max(loss, torch.zeros_like(loss))

        if self.__smoothness_constraint:
            loss += torch.sum((anomaly_preds[:, :-1] - anomaly_preds[:, 1:]) ** 2) * self.__smoothness_weight

        if self.__sparsity_constraint:
            loss += torch.sum(anomaly_preds, dim=1) * self.__sparsity_weight

        loss = torch.sum(loss).mean(0) if self.__reduction == "mean" else torch.sum(loss)
        return loss
