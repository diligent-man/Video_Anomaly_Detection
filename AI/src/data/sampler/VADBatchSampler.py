from typing import List, Union, Iterable, Iterator
from torch.utils.data.sampler import BatchSampler, Sampler


__all__ = ["VADBatchSampler"]


class VADBatchSampler(BatchSampler):
    def __init__(self,
                 sampler: Union[Sampler[int], Iterable[int]],
                 batch_size: int,
                 ) -> None:
        super(VADBatchSampler, self).__init__(sampler, batch_size, False)

    def __iter__(self) -> Iterator[List[int]]:
        """
        :return:  iterator of normal and anomalous indices
        """
        batch: List[List[int]] = [[0] * self.batch_size, [0] * self.batch_size]

        idx_in_batch = 0
        for (idx1, idx2) in self.sampler:
            batch[0][idx_in_batch], batch[1][idx_in_batch] = idx1, idx2
            idx_in_batch += 1

            if idx_in_batch % self.batch_size == 0:
                yield zip(batch[0], batch[1])

                idx_in_batch = 0
                batch = [[0] * self.batch_size, [0] * self.batch_size]

        if idx_in_batch > 0:
            yield zip(batch[0], batch[1])
