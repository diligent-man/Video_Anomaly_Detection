from typing import Tuple, List, Iterator

import torch
from torch.utils.data.sampler import Sampler

from ..dataset import VADVideoLevelDataset


__all__ = ["VADSampler"]


class VADSampler(Sampler):
    """
    Sampler for video anomaly detection. Dataset folder is placed as below:
        unlabeled
            anomaly
                class 1
                    *.mp4
                class 2
                    *.mp4
                ...
            normal
                *.mp4

    Two approaches for sampling:
        1/ For each epoch,
            randomly select a number of samples from the larger datasource with length equivalent to the smaller one.

        2/ For each epoch,
            Padding the smaller dataset with additional randomly selected samples from its
    """
    def __init__(self, data_source: VADVideoLevelDataset, method: int, shuffle: bool) -> None:
        """
        :param data_source: contains normal (VideoDataset) and anomaly (VideoFolderDataset) datasource
        :param method:
            - 1: cut samples to the smaller one
            - 2: pad samples to the larger one
        :param shuffle: shuffle indices or not
        """

        super(VADSampler).__init__()

        self.__method: int = method
        self.__shuffle: bool = shuffle
        self.__data_source: VADVideoLevelDataset = data_source

    def __len__(self) -> int:
        data_len: int = min(self.__data_source.__len__()) if self.__method == 1 else max(self.__data_source.__len__())
        return data_len

    def __iter__(self) -> Tuple[Iterator[int], Iterator[int]]:
        min_source_len: int = min(self.__data_source.__len__())
        max_source_len: int = max(self.__data_source.__len__())

        if self.__method == 1:
            first_source_indices: List[int] = torch.randperm(min_source_len).tolist()
            second_source_indices: List[int] = torch.randperm(max_source_len)[: min_source_len].tolist()
        else:
            first_source_indices: List[int] = torch.cat((
                torch.randperm(min_source_len),
                torch.randperm(min_source_len)[: max_source_len-min_source_len]
            ), 0).tolist()

            second_source_indices: List[int] = torch.randperm(max_source_len).tolist()

        if not self.__shuffle:
            first_source_indices = sorted(first_source_indices)
            second_source_indices = sorted(second_source_indices)
        return zip(iter(first_source_indices), iter(second_source_indices))
