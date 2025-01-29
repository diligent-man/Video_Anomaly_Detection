from typing import List

import torch
import overrides

from bisect import bisect_right


__all__ = ["CustomSequentialLR"]


class CustomSequentialLR(torch.optim.lr_scheduler.SequentialLR):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        schedulers: List[torch.optim.lr_scheduler.LRScheduler],
        milestones: List[int],
        last_epoch=-1,
        verbose="deprecated",
    ) -> None:
        super().__init__(optimizer, schedulers, milestones, last_epoch, verbose)

    @overrides.override
    def step(self) -> None:
        """Perform a step."""
        self.last_epoch += 1
        idx = bisect_right(self._milestones, self.last_epoch)
        scheduler: torch.optim.lr_scheduler.LRScheduler = self._schedulers[idx]

        scheduler.step()

        self._last_lr = scheduler.get_last_lr()
