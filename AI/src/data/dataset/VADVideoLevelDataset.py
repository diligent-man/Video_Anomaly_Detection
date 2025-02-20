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
    def __init__(self,
                 normal_root: str,
                 anomaly_root: str,
                 loader: str,
                 return_device: str = "cpu",
    ) -> None:
        self.normal_ds: VideoDataset = VideoDataset(normal_root, loader, target=0)
        self.anomaly_ds: VideoFolderDataset = VideoFolderDataset(anomaly_root, loader)

    def __len__(self) -> Tuple[int, int]:
        return len(self.normal_ds), len(self.anomaly_ds)

    def __getitem__(self, indices: Tuple[int, int]) -> Tuple[Tensor, Tensor]:
        """
        :param indices:
            - 1st idx: idx from iterator with smaller length dataset in both 2 sampling methods of VADSampler
            - 2nd idx: idx from iterator with larger length dataset in both 2 sampling methods of VADSampler
        :return: (anomaly_idx, normal_idx)
        """
        if len(self.normal_ds) <= len(self.anomaly_ds):
            normal_idx, anomaly_idx = indices
        else:
            normal_idx, anomaly_idx = reversed(indices)
        return torch.rand(3, 224, 224), torch.rand(3, 224, 224)


from AI.src.data.dataloader import VADVideoLevelDataLoader


def main() -> None:
    ds = VADVideoLevelDataset(
        normal_root="/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/dataset/ucf-test/unlabeled/normal",
        anomaly_root="/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/dataset/ucf-test/unlabeled/anomaly",
        loader="v3"
    )
    dl = VADVideoLevelDataLoader(ds, 4, True, 1, num_workers=4)

    # print(dl)
    for i, (normal, anomaly) in enumerate(dl):
        print(i, normal.shape, anomaly.shape)
        break
    return None


if __name__ == '__main__':
    main()