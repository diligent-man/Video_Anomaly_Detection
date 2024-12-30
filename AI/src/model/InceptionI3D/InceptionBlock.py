from typing import List

import torch

from .Unit3D import Unit3D
from .MaxPool3dSamePadding import MaxPool3dSamePadding

__all__ = ["InceptionBlock"]


class InceptionBlock(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: List[int],
                 name: str
                 ):
        super(InceptionBlock, self).__init__()
        self.b0 = Unit3D(in_channels, out_channels[0], name=name + "/Branch_0/Conv3d_0a_1x1")

        self.b1a = Unit3D(in_channels, out_channels[1], name=name + "/Branch_1/Conv3d_0a_1x1")
        self.b1b = Unit3D(out_channels[1], out_channels[2], (3, 3, 3), name=name + '/Branch_1/Conv3d_0b_3x3')

        self.b2a = Unit3D(in_channels, out_channels[3], name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(out_channels[3], out_channels[4], (3, 3, 3), name=name + '/Branch_2/Conv3d_0b_3x3')

        self.b3a = MaxPool3dSamePadding()
        self.b3b = Unit3D(in_channels, out_channels[5], name=name + '/Branch_3/Conv3d_0b_1x1')

        self.name = name

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)
