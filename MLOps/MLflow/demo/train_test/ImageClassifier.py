"""
Quick simple user-defined pytorch model
Dataset: MNIST
Num classes: 10
"""
import torch

__all__ = ["ImageClassifier"]


class ImageClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(1, 8, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Conv2d(8, 16, kernel_size=3),
            torch.nn.ReLU(),
            torch.nn.Flatten(),
            torch.nn.Linear(in_features=9216, out_features=10)
        )

    def forward(self, x):
        return self.model(x)
