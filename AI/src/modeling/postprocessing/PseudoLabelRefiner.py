from typing import Tuple

import torch

from torch import Tensor
from torch.nn import Module, AvgPool1d, ConstantPad1d


__all__ = ["PseudoLabelRefiner"]


class PseudoLabelRefiner(Module):
    """
    Refine mode's output by 2 stages:
        a/ Moving average with mean EMA padding
        b/ Min-max norm

    Example:
        input shape: 32 segments. Each segment is a vector representing 30 frames of video having 15 FPS
        kernel size: 3
        pad: True
        => Non-padded regions:
            [0,14] -> [17,31]
                => 18
           Padded regions:
                r1: [-7,7] -> [0,14]
                    => 7
                r2: [17,31] -> [23,38]
                    => 7
        Padded value is mean of values at index from 0 to kernel_size // 2
    """
    __PAD_STYLES = ("zero", "mean_ema")

    def __init__(self,
                 eps: float = 1e-8,
                 weight: float = 0.8,
                 kernel_size: int = 3,
                 pad_style: str = "zero"
                 ) -> None:
        """
        :param kernel_size: Size of the moving average filter
        :param eps: Small epsilon to avoid division by zero in normalization
        """
        assert pad_style in self.__PAD_STYLES, ValueError(f"Currently support {self.__PAD_STYLES}")

        super(PseudoLabelRefiner, self).__init__()
        self.__eps: float = eps
        self.__weight: float = weight
        self.__kernel_size: int = kernel_size
        self.__pad_style: str = pad_style
        self.__filter: AvgPool1d = AvgPool1d(kernel_size, 1)

    def _compute_pad_size(self) -> Tuple[int, int]:
        pad_size: int = max(int(self.__kernel_size) - 1, 0)

        left_pad: int = pad_size // 2
        right_pad: int = pad_size - left_pad
        return left_pad, right_pad

    def _pad_mean_ema(self, x: Tensor) -> Tensor:
        """
        :param x: input tensor with shape (B, T)
        :return:
        """
        left_pad, right_pad = self._compute_pad_size()
        left_weight: Tensor = torch.tensor([self.__weight ** i for i in reversed(range(1, left_pad + 1))], device=x.device)
        right_weight: Tensor = torch.tensor([self.__weight ** i for i in range(1, right_pad + 1)], device=x.device)

        x: Tensor = ConstantPad1d((left_pad, right_pad), 0)(x)
        x[:, :left_pad] = x[:, left_pad: 2*left_pad].mean(-1, keepdim=True) * left_weight
        x[:, -right_pad:] = x[:, -(2*right_pad): -right_pad].mean(-1, keepdim=True) * right_weight
        return x

    def _min_max_norm(self, x: Tensor) -> Tensor:
        min_val: Tensor = x.min(-1, True).values
        max_val: Tensor = x.max(-1, True).values
        return (x - min_val) / (max_val - min_val + self.__eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Refines pseudo-labels using moving average and min-max normalization.

        :param x: anomalous scores for each video segment. Shape [batch_size, num_segments]
        :return: Refined pseudo-label for each video segment
        """
        assert x.dim() == 2, ValueError

        if self.__pad_style == "zero":
            self.__filter = AvgPool1d(self.__kernel_size, 1, self.__kernel_size//2)
        else:
            x = self._pad_mean_ema(x)

        x = self.__filter(x)
        x = self._min_max_norm(x)
        return x
