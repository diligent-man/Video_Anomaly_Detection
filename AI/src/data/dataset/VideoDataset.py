import os
import torch
import inspect
import functools

from typing import Callable, Optional, Tuple, List, Any, Dict

from torch.utils.data import Dataset

from ...utils import video_loader


__all__ = ["VideoDataset"]


class VideoDataset(Dataset):
    """
    Used in conjunction with DataLoader for batch loading and further processing steps.
    """
    _repr_indent = 4

    def __init__(self,
                 root: str,
                 loader: str = "v2",
                 loader_args: Optional[Dict[str, Any]] = None,
                 extensions: Optional[Tuple[str, ...]] = ("mp4", "avi"),
                 transforms: Optional[Callable] = None,
                 target_transforms: Optional[Callable] = None,
                 device: str = "cpu",
                 return_device: str = "cpu",
                 target: Optional[Any] = None
                 ) -> None:
        """
        :param root: dir of videos
        :param loader: video loader api. Defaults to "v2"
        :param loader_args: arguments for video loader
        :param extensions: video extension
        :param transforms: transform function for input video
        :param target_transforms: transform function for label
        :param device: device that used to load video
        :param return_device: device that used to return read video
        :param target: target for input video if necessary
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

        self.__root: str = root
        self.__loader: Callable = functools.partial(loader, **loader_args)
        self.__transforms: Optional[Callable] = transforms
        self.__target_transforms: Optional[Callable] = target_transforms
        self.__return_device: str = return_device
        self.__target: Optional[Any] = target

    @staticmethod
    def _extra_repr() -> str:
        return ""

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, Any]:
        video_path: str = os.path.join(self.__root, os.listdir(self.__root)[index])
        sample: torch.Tensor = self.__loader(video_path)

        if self.__transforms is not None:
            sample: torch.Tensor = self.__transforms(sample)

        if self.__target_transforms is not None:
            target: torch.Tensor = self.__target_transforms(self.__target)
        else:
            target: Any = self.__target

        # sample = sample[:200, ...]  # temporary add for loading
        return sample.to(self.__return_device), target

    def __len__(self) -> int:
        return len(os.listdir(self.__root))

    def __repr__(self) -> str:
        head: str = "Dataset " + self.__class__.__name__
        body: List[str] = [f"Number of datapoints: {self.__len__()}"]

        if self.__root is not None:
            body.append(f"Root location: {self.__root}")

        body += self._extra_repr().splitlines()

        if hasattr(self, "transforms") and self.__transforms is not None:
            body += [repr(self.transforms)]

        lines = [head] + [" " * self._repr_indent + line for line in body]
        return "\n".join(lines)
