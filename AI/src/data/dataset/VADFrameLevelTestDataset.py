import os
import inspect
import functools

from typing import Optional, Callable, Any, Tuple, Dict, List


import torch
import numpy as np
import pandas as pd
from torchvision.datasets import VisionDataset


from ...utils import video_loader


__all__ = ["VADFrameLevelTestDataset"]


class VADFrameLevelTestDataset(VisionDataset):
    _repr_indent = 4

    def __init__(self,
                 root: str,
                 annotation: str,
                 loader: str = "v4",
                 loader_args: Optional[Dict[str, Any]] = None,
                 extensions: Optional[Tuple[str, ...]] = ("mp4", "avi", "pt"),
                 target_transform: Optional[Callable] = None,
                 device: str = "cpu",
                 return_device: str = "cpu",
                 ) -> None:
        assert os.path.isdir(root), NotADirectoryError
        assert loader in video_loader.keys(), NotImplementedError
        assert set(extensions) <= {"mp4", "avi", "pt"}, "Currently only supports mp4 video"

        super(VADFrameLevelTestDataset, self).__init__(root, None, None, target_transform)
        loader: Callable = video_loader[loader]

        if loader_args is None:
            loader_args = {}

        if "device" in inspect.signature(loader).parameters:
            loader_args = {"device": device, **loader_args}

        self.__root: str = root
        self.__loader: Callable = functools.partial(loader, **loader_args)
        self.__annotation_fname: str = annotation
        self.__annotation: pd.DataFrame = pd.read_csv(
            os.path.join(root, annotation),
            header=None,
            names=["path", "start1", "end1", "start2", "end2"]
        )
        self.__return_device: str = return_device

    @staticmethod
    def extra_repr(*_) -> str:
        return ""

    def __len__(self) -> int:
        return len(self.__annotation)

    def __getitem__(self, idx: int) -> Tuple[int, str, torch.Tensor]:
        fpath: str = self.__annotation["path"][idx]
        fpath: str = os.path.join(self.__root, fpath)

        frames: int = self.__loader(fpath).shape[0]  # [T, H, W, C]
        labels: torch.Tensor = torch.zeros((frames, ), dtype=torch.uint8, device=self.__return_device)
        labeled_indices: List[np.int64] = self.__annotation.iloc[idx, self.__annotation.columns != "path"].to_list()
        for i in range(0, len(labeled_indices), 2):
            start, end = labeled_indices[i: i + 2]
            start, end = int(float(start)), int(float(end))

            if (start, end) != (-1, -1):
                labels[start: end + 1] = 1

        if self.target_transform is not None:
            labels = self.target_transform(labels)
        return idx, fpath, labels

    def __repr__(self):
        num_anomaly: int = len(self.__annotation.loc[self.__annotation.path.str.startswith('anomaly')])
        num_normal: int = len(self.__annotation.loc[self.__annotation.path.str.startswith('normal')])

        head = "Dataset " + self.__class__.__name__ + " includes:"

        body = [f"Number of datapoints: {self.__len__()} ({num_anomaly} anomaly, {num_normal} normal)"]
        body += [f"Root location: {self.__root}"]
        body += [f"Annotation: {self.__annotation_fname}"]

        body += self.extra_repr().splitlines()

        if hasattr(self, "transform") and self.transform is not None:
            body += self._format_transform_repr(self.transform, 'Transform: ')

        if hasattr(self, "target_transform") and self.target_transform is not None:
            body += self._format_transform_repr(self.target_transform, 'Target transform: ')

        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)
