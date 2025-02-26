import os
import inspect
import functools
from typing import Optional, Callable, Any, Tuple, Dict, List

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ...utils import video_loader


__all__ = ["VADFrameLevelDataset"]


class VADFrameLevelDataset(Dataset):
    def __init__(self,
                 root: str,
                 annotation: str,
                 loader: str = "v2",
                 loader_args: Optional[Dict[str, Any]] = None,
                 extensions: Optional[Tuple[str, ...]] = ("mp4", "avi", "pt"),
                 transforms: Optional[Callable] = None,
                 target_transforms: Optional[Callable] = None,
                 device: str = "cpu",
                 return_device: str = "cpu",
                 ) -> None:
        assert os.path.isdir(root), NotADirectoryError
        assert loader in video_loader.keys(), NotImplementedError
        assert set(extensions) <= {"mp4", "avi", "pt"}, "Currently only supports mp4 video"

        loader: Callable = video_loader[loader]

        if loader_args is None:
            loader_args = {}

        if "device" in inspect.signature(loader).parameters:
            loader_args = {"device": device, **loader_args}

        self.__root: str = root
        self.__loader: Callable = functools.partial(loader, **loader_args)
        self.__annotation: pd.DataFrame = pd.read_csv(os.path.join(root, annotation))
        self.__transforms: Optional[Callable] = transforms
        self.__target_transforms: Optional[Callable] = target_transforms
        self.__return_device: str = return_device

    def __len__(self):
        return len(self.__annotation)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        fpath: str = self.__annotation["path"][idx]
        fpath: str = os.path.join(self.__root, fpath)

        frames: torch.Tensor = self.__loader(fpath)  # [T, H, W, C]
        frames = frames.permute(0, -1, 1, 2)

        labels: torch.Tensor = torch.zeros(frames.shape[0], dtype=torch.uint8, device=frames.device)
        labeled_indices: List[np.int64] = self.__annotation.iloc[idx, self.__annotation.columns != "path"].to_list()
        for i in range(0, len(labeled_indices), 2):
            start, end = labeled_indices[i: i+2]

            if (start, end) != (-1, -1):
                labels[start: end+1] = 1

        if self.__transforms is not None:
            frames = self.__transforms(frames)

        if self.__target_transforms is not None:
            labels = self.__target_transforms(labels)
        return frames, labels
