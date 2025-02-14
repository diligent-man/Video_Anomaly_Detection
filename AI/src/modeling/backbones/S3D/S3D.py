from typing import Callable, Optional, Any
from functools import partial

import torch
from torchvision.ops import Conv3dNormActivation

from ....utils import load_weights
from .S3D_Weights import S3D_Weights
from .TemporalSeparableConv import TemporalSeparableConv
from .SepInceptionBlock3D import SepInceptionBlock3D

__all__ = ["s3d"]


class S3D(torch.nn.Module):
    def __init__(
            self,
            num_classes: int = 400,
            dropout: float = 0.2,
            norm_layer: Optional[Callable[..., torch.nn.Module]] = None,
    ) -> None:
        """
        :param num_class (int): number of classes for the classification task.
        :param dropout (float): dropout probability.
        :param norm_layer (Optional[Callable]): Module specifying the normalization layer to use.
        """
        super(S3D, self).__init__()

        if norm_layer is None:
            norm_layer = partial(torch.nn.BatchNorm3d, eps=0.001, momentum=0.001)

        self.features = torch.nn.Sequential(
            TemporalSeparableConv(3, 64, 7, 2, 3, norm_layer),
            torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            Conv3dNormActivation(
                64,
                64,
                kernel_size=1,
                stride=1,
                norm_layer=norm_layer,
            ),
            TemporalSeparableConv(64, 192, 3, 1, 1, norm_layer),
            torch.nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)),
            SepInceptionBlock3D(192, 64, 96, 128, 16, 32, 32, norm_layer),
            SepInceptionBlock3D(256, 128, 128, 192, 32, 96, 64, norm_layer),
            torch.nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1)),
            SepInceptionBlock3D(480, 192, 96, 208, 16, 48, 64, norm_layer),
            SepInceptionBlock3D(512, 160, 112, 224, 24, 64, 64, norm_layer),
            SepInceptionBlock3D(512, 128, 128, 256, 24, 64, 64, norm_layer),
            SepInceptionBlock3D(512, 112, 144, 288, 32, 64, 64, norm_layer),
            SepInceptionBlock3D(528, 256, 160, 320, 32, 128, 128, norm_layer),
            torch.nn.MaxPool3d(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding=(0, 0, 0)),
            SepInceptionBlock3D(832, 256, 160, 320, 32, 128, 128, norm_layer),
            SepInceptionBlock3D(832, 384, 192, 384, 48, 128, 128, norm_layer),
        )
        self.avgpool = torch.nn.AvgPool3d(kernel_size=(2, 7, 7), stride=1)
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(p=dropout),
            torch.nn.Conv3d(1024, num_classes, kernel_size=1, stride=1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: batch of videos with dimensions (batch, channel, time, height, width)
        :return: torch.Tensor
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = self.classifier(x)
        x = torch.mean(x, dim=(2, 3, 4))
        return x


def s3d(weights: None | str | S3D_Weights = True, **kwargs) -> torch.nn.Module:
    model: S3D = S3D(**kwargs)
    weights: None | dict[str, Any] = load_weights(weights)

    if weights is not None:
        model.load_state_dict(weights)
    return model

