import torch

__all__ = ["PseudoLabelRefiner"]


class PseudoLabelRefiner(torch.nn.Module):
    __eps: float
    __MA_filter: torch.nn.AvgPool1d

    def __init__(self,
                 kernel_size: int = 3,
                 eps: float = 1e-8
                 ) -> None:
        """
        :param kernel_size: Size of the moving average filter
        :param eps: Small epsilon to avoid division by zero in normalization
        """
        super(PseudoLabelRefiner, self).__init__()
        self.__eps = eps
        self.__filter = torch.nn.AvgPool1d(kernel_size, 1, kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Refines pseudo-labels using moving average and min-max normalization.

        :param x: anomalous scores for each video segment. Shape [batch_size, num_segments, 1]
        :return: Refined pseudo-labels. Shape [batch_size, num_segments, num_classes=1]
        """
        x = self.__filter(x)
        x = (x - x.min()) / (x.max() - x.min() + self.__eps)
        return x
