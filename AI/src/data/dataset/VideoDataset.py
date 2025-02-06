import os
import torch
import inspect
import functools

from ...utils import load_video_v2
from typing import Callable, Optional, Tuple, List


__all__ = ["VideoDataset"]


class VideoDataset(torch.utils.data.Dataset):
    """
    Used in conjunction with torch.utils.data.Dataloader for batch loading and further processing steps.
    """
    _repr_indent = 4

    def __init__(self,
                 root: str,
                 extensions: Optional[Tuple[str, ...]],
                 device: str = "cpu",
                 return_device: str = "cpu",
                 transforms: Optional[Callable] = None,
                 loader: Callable[[str], ...] = load_video_v2,
                 **kwargs
                 ) -> None:
        """
        :param root: dir of videos
        :param extensions: tested with mp4
        :param device: device that used to load video
        :param return_device: device that used to return read video
        :param transforms: transform function
        :param loader: video loader function
        """
        assert extensions in ["mp4"], "Currently only supports mp4 video"
        assert os.path.isdir(root), NotADirectoryError

        if "device" in inspect.signature(loader).parameters:
            kwargs = {"device": device, **kwargs}

        self.__root: root = root
        self.__extensions: Tuple[str, ...] = extensions
        self.__device: str = device
        self.__return_device: str = return_device
        self.__transforms: Callable = transforms
        self.__loader: Callable = functools.partial(loader, **kwargs)

    @staticmethod
    def _extra_repr() -> str:
        return ""

    def __getitem__(self, index: int) -> Tuple[str, torch.Tensor]:
        video_path = os.path.join(self.__root, os.listdir(self.__root)[index])
        sample: torch.Tensor = self.__loader(video_path)

        if self.__transforms is not None:
            sample = self.__transforms(sample)

        sample = sample[:200, ...]
        return video_path, sample.to(self.__return_device)

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
