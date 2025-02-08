import torch
import torchvision

from pathlib import Path
from typing import Union, Callable, Any, Optional, Tuple

from ...utils import video_loader


__all__ = ["VideoFolderDataset"]


class VideoFolderDataset(torchvision.datasets.DatasetFolder):
    def __init__(self,
                 root: Union[str, Path],
                 loader: str = "v2",
                 extensions: Optional[Tuple[str, ...]] = "mp4",
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 is_valid_file: Optional[Callable[[str], bool]] = None,
                 allow_empty: bool = False
                 ):
        assert loader in video_loader.keys(), ValueError(f"Unsupported video loader. Currently {video_loader.keys()}")
        loader: Callable[[str], torch.Tensor] = video_loader[loader]

        super().__init__(
            root,
            loader,
            extensions,
            transform,
            target_transform,
            is_valid_file,
            allow_empty
        )

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
        return sample, target

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
