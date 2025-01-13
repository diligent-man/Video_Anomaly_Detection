import torch

__all__ = ["PseudoLabelRefiner"]


class PseudoLabelRefiner(object):
    __window_size: int
    __eps: float

    def __init__(self,
                 window_size: int = 3,
                 eps: float = 1e-8
                 ) -> None:
        """
        :param window_size: Size of the moving average filter
        :param eps: Small epsilon to avoid division by zero in normalization
        """
        super(PseudoLabelRefiner, self).__init__()
        self.__window_size = window_size
        self.__eps = eps
    
    def refine(self, anomaly_scores: torch.Tensor) -> torch.Tensor:
        """
        Refines pseudo-labels using moving average and min-max normalization.

        :param anomaly_scores: preds for video segments from model. Shape [batch_size, num_segments, num_classes]
        :return: Refined pseudo-labels. Shape [batch_size, num_segments, num_classes]
        """
        # Apply moving average filter
        smoothed_scores = torch.nn.functional.avg_pool1d(
            anomaly_scores.unsqueeze(1),  # Add channel dimension
            kernel_size=self.__window_size,
            stride=1,
            padding=self.__window_size // 2  # Same padding
        ).squeeze(1)  # Remove channel dimension

        # Apply min-max normalization
        min_scores = smoothed_scores.min(dim=1, keepdim=True)[0]  # Min along segments
        max_scores = smoothed_scores.max(dim=1, keepdim=True)[0]  # Max along segments

        refined_scores = (smoothed_scores - min_scores) / (max_scores - min_scores + self.__eps)
        return refined_scores

    def __call__(self, anomaly_scores: torch.Tensor) -> torch.Tensor:
        refined_scores = self.refine(anomaly_scores)
        return refined_scores
