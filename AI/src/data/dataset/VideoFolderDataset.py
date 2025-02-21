import os
import inspect

from pathlib import Path
from functools import partial
from typing import Union, Callable, Any, Optional, Tuple, Dict


import torch
from torchvision.datasets import DatasetFolder


from ...utils import video_loader


__all__ = ["VideoFolderDataset"]


class VideoFolderDataset(DatasetFolder):
    def __init__(self,
                 root: Union[str, Path],
                 loader: str = "v2",
                 loader_args: Optional[Dict[str, Any]] = None,
                 extensions: Optional[Tuple[str, ...]] = ("mp4", "avi"),
                 transforms: Optional[Callable] = None,
                 target_transforms: Optional[Callable] = None,
                 device: str = "cpu",
                 return_device: str = "cpu"
                 ):
        """
        :param root: dir of videos
        :param loader: video loader api. Defaults to "v2"
        :param loader_args: arguments for video loader
        :param extensions: video extension
        :param transforms: transform function for input video
        :param target_transforms: transform function for label
        :param device: device that used to load video
        :param return_device: device that used to return read video
        """
        assert os.path.isdir(root), NotADirectoryError
        assert loader in video_loader.keys(), NotImplementedError
        assert set(extensions) <= {"mp4", "avi"}, "Currently only supports mp4 video"
        assert device in ("cpu", "cuda"), "Currently only supports cpu/ cuda device"

        loader: Callable = video_loader[loader]

        if loader_args is None:
            loader_args = {}

        if "device" in inspect.signature(loader).parameters:
            loader_args = {"device": device, **loader_args}

        super().__init__(
            root,
            partial(loader, **loader_args),
            extensions,
            transforms,
            target_transforms,
            None,
            True
        )

        self.__transforms: Optional[Callable] = transforms
        self.__target_transforms: Optional[Callable] = target_transforms
        self.__return_device: str = return_device

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample: torch.Tensor = self.loader(path)

        if self.transform is not None:
            sample = self.transform(sample)

        if self.target_transform is not None:
            target = self.target_transform(target)

        # sample = sample[:200, ...]  # temporary add for loading
        return sample.to(self.__return_device), target

    def __len__(self) -> int:
        return len(self.samples)

    def __repr__(self) -> str:
        head = "Dataset " + self.__class__.__name__
        body = [f"Number of datapoints: {self.__len__()}"]

        if self.root is not None:
            body.append(f"Root location: {self.root}")
        body += self.extra_repr().splitlines()

        if hasattr(self, "transforms") and self.transforms is not None:
            body += [repr(self.transforms)]
        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)
