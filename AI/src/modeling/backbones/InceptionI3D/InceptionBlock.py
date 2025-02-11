from typing import List

import torch

from .Unit3D import Unit3D
from .MaxPool3dSamePadding import MaxPool3dSamePadding

__all__ = ["InceptionBlock"]


class InceptionBlock(torch.nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: List[int],
                 ):
        super(InceptionBlock, self).__init__()
        b0 = Unit3D(in_channels, out_channels[0])

        b1a = Unit3D(in_channels, out_channels[1])
        b1b = Unit3D(out_channels[1], out_channels[2], (3, 3, 3))

        b2a = Unit3D(in_channels, out_channels[3])
        b2b = Unit3D(out_channels[3], out_channels[4], (3, 3, 3))

        b3a = MaxPool3dSamePadding()
        b3b = Unit3D(in_channels, out_channels[5])

        self.branch_0 = b0
        self.branch_1 = torch.nn.Sequential(b1a, b1b)
        self.branch_2 = torch.nn.Sequential(b2a, b2b)
        self.branch_3 = torch.nn.Sequential(b3a, b3b)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_0_out = self.branch_0(x)
        branch_1_out = self.branch_1(x)
        branch_2_out = self.branch_2(x)
        branch_3_out = self.branch_3(x)
        return torch.cat([branch_0_out, branch_1_out, branch_2_out, branch_3_out], 1)
