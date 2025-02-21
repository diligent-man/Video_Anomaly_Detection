import os
import inspect
import functools

from typing import Optional, Callable, Any, Tuple, Dict


import torch
import pandas as pd

from torch.utils.data import Dataset


from AI.src.utils import video_loader


__all__ = ["VADFrameLevelDataset"]


class VADFrameLevelDataset(Dataset):
    def __init__(self,
                 root: str,
                 loader: str = "v2",
                 loader_args: Optional[Dict[str, Any]] = None,
                 extensions: Optional[Tuple[str, ...]] = ("mp4", "avi"),
                 annotation_fname: str = "label.csv",
                 transforms: Optional[Callable] = None,
                 target_transforms: Optional[Callable] = None,
                 device: str = "cpu",
                 return_device: str = "cpu",
                 ) -> None:
        assert os.path.isdir(root), NotADirectoryError
        assert loader in video_loader.keys(), NotImplementedError
        assert set(extensions) <= {"mp4", "avi"}, "Currently only supports mp4 video"

        loader: Callable = video_loader[loader]

        if loader_args is None:
            loader_args = {}

        if "device" in inspect.signature(loader).parameters:
            loader_args = {"device": device, **loader_args}

        self.__root: str = root
        self.__loader: Callable = functools.partial(loader, **loader_args)
        self.__annotation: pd.DataFrame = pd.read_csv(os.path.join(root, annotation_fname))
        self.__transforms: Optional[Callable] = transforms
        self.__target_transforms: Optional[Callable] = target_transforms
        self.__return_device: str = return_device

    def __len__(self):
        return len(self.__annotation)

    def __getitem__(self, idx: int):
        fpath = self.__annotation["path"][idx]
        # fpath = os.path.join("anomaly", fpath) if fpath.startswith("anomaly") else os.path.join("normal", fpath)
        fpath: str = os.path.join(self.__root, self.__annotation["class"][idx], fpath)
        print(fpath)


        inp: torch.Tensor = self.__loader(fpath)
        # target: torch.Tensor =

from AI.src.data.dataloader import DefaultDataLoader


def main() -> None:
    ds = VADFrameLevelDataset(
        "../../../dataset/ucf-test/labeled",
        "v5",
        annotation_fname="label.csv",
    )

    dl = DefaultDataLoader(ds, 1, False)
    # print(ds)
    # inps, labels = next(iter(dl))
    # print(inps.shape, labels.shape)
    next(dl.__iter__())
    return None


if __name__ == '__main__':
    main()
