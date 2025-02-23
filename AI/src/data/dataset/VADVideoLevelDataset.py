from typing import Tuple

import torch
from torch import Tensor
from torch.utils.data import Dataset

from AI.src.data.dataset.VideoDataset import VideoDataset
from AI.src.data.dataset.VideoFolderDataset import VideoFolderDataset


__all__ = ["VADVideoLevelDataset"]


class VADVideoLevelDataset(Dataset):
    """
    Simultaneously read anomalous and abnormal video at video-level label.
    """
    _repr_indent: int = 4

    def __init__(self,
                 normal_root: str,
                 anomaly_root: str,
                 loader: str,
                 **kwargs
                 ) -> None:
        self.normal_ds: VideoDataset = VideoDataset(normal_root, loader, target=0, **kwargs)
        self.anomaly_ds: VideoFolderDataset = VideoFolderDataset(anomaly_root, loader, **kwargs)

    @staticmethod
    def extra_repr() -> str:
        return ""

    def __len__(self) -> Tuple[int, int]:
        return len(self.normal_ds), len(self.anomaly_ds)

    def __getitem__(self,
                    indices: Tuple[int, int]
                    ) -> Tuple[Tensor, Tensor]:
        """
        :param indices:
            - 1st idx:
            idx from iterator with smaller length dataset in both sampling methods of VADSampler
            - 2nd idx:
            idx from iterator with larger length dataset in both sampling methods of VADSampler
        :return: Tuple of inputs and targets.
            Inputs shape: (B, 64, C, T, H, W) with v4 video loader
            Targets shape: (B, 2)
        """
        if len(self.anomaly_ds) <= len(self.normal_ds):
            anomaly_idx, normal_idx = indices
        else:
            anomaly_idx, normal_idx = reversed(indices)

        anomaly, _ = self.anomaly_ds.__getitem__(anomaly_idx)
        normal, _ = self.normal_ds.__getitem__(normal_idx)
        return torch.vstack((anomaly, normal)), torch.tensor(1, )

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__ + " includes:"

        body = [f"{self.normal_ds.__repr__()}\n\n\t{self.anomaly_ds.__repr__()}"]
        body += self.extra_repr().splitlines()

        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)
