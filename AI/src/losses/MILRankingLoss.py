import torch


__all__ = ["MILRankingLoss"]


class MILRankingLoss(torch.nn.MarginRankingLoss):
    """
    Ref: Real-world Anomaly Detection in Surveillance Videos
    https://github.com/seominseok0429/Real-world-Anomaly-Detection-in-Surveillance-Videos-pytorch/blob/main/loss.py
    """
    def __init__(self,
                 margin=0.0,
                 sparsity_constraint: bool = False,
                 smoothness_constraint: bool = False,
                 size_average=None,
                 reduce=None,
                 reduction='mean',
                 ) -> None:
        super(MILRankingLoss, self).__init__(margin, size_average, reduce, reduction)
        self.__sparsity_constraint = sparsity_constraint
        self.__smoothness_constraint = smoothness_constraint

    def forward(self, input1: torch.Tensor, input2: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        :param input1: 0D or 1D tensor
        :param input2: 0D or 1D tensor
        :param target: 0D or 1D tensor
        :return: scalar
        """
        loss: torch.Tensor = super()(input1, input2, target)
        # return F.margin_ranking_loss(
        #     input1, input2, target, margin=self.margin, reduction=self.reduction
        # )
        return loss
