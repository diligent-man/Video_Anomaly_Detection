import torch

__all__ = ["PseudoLabelRefiner"]


class PseudoLabelRefiner:
    def __init__(self, window_size=3, eps=1e-8):
        """
        Initializes the PseudoLabelRefiner.

        Parameters:
        - window_size (int): Size of the moving average filter (default: 3).
        - eps (float): Small epsilon to avoid division by zero in normalization (default: 1e-8).
        """
        self.window_size = window_size
        self.eps = eps
    
    def refine(self, anomaly_scores):
        """
        Refines pseudo-labels using moving average and min-max normalization.

        Parameters:
        - anomaly_scores (torch.Tensor): Tensor of anomaly scores [batch_size, num_segments].

        Returns:
        - refined_scores (torch.Tensor): Refined pseudo-labels [batch_size, num_segments].
        """

        # Apply moving average filter
        smoothed_scores = torch.nn.functional.avg_pool1d(
            anomaly_scores.unsqueeze(1),  # Add channel dimension
            kernel_size=self.window_size,
            stride=1,
            padding=self.window_size // 2  # Same padding
        ).squeeze(1)  # Remove channel dimension

        # Apply min-max normalization
        min_scores = smoothed_scores.min(dim=1, keepdim=True)[0]  # Min along segments
        max_scores = smoothed_scores.max(dim=1, keepdim=True)[0]  # Max along segments

        refined_scores = (smoothed_scores - min_scores) / (max_scores - min_scores + self.eps)

        return refined_scores
    

    def __call__(self, anomaly_scores):
        """
        Enables the class instance to be called like a function.

        Parameters:
        - anomaly_scores (torch.Tensor): Tensor of anomaly scores [batch_size, num_segments].

        Returns:
        - refined_scores (torch.Tensor): Refined pseudo-labels [batch_size, num_segments].
        """
        return self.refine(anomaly_scores)
    

# # Giả lập anomaly scores
# batch_size = 4
# num_segments = 10
# torch.manual_seed(42)

# # Tạo điểm bất thường ngẫu nhiên
# anomaly_scores = torch.rand(batch_size, num_segments)
# print("Original Anomaly Scores:")
# print(anomaly_scores)

# # Khởi tạo refiner
# refiner = PseudoLabelRefiner(window_size=3)

# # Áp dụng pseudo-label refinement
# refined_scores = refiner(anomaly_scores)
# print("\nRefined Pseudo-Labels:")
# print(refined_scores)