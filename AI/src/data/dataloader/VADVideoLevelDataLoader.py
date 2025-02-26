from torch.utils.data import DataLoader

from ..dataset import VADVideoLevelDataset
from ..sampler import VADSampler, VADBatchSampler


__all__ = ["VADVideoLevelDataLoader"]


class VADVideoLevelDataLoader(DataLoader):
    def __init__(self,
                 dataset: VADVideoLevelDataset,
                 batch_size: int,
                 shuffle: bool,
                 method: int,
                 **kwargs
                 ) -> None:
        self.__dataset = dataset
        super(VADVideoLevelDataLoader, self).__init__(
            dataset,
            batch_sampler=VADBatchSampler(VADSampler(dataset, method, shuffle), batch_size),
            **kwargs
        )

    def __repr__(self):
        return self.__dataset.__repr__()
