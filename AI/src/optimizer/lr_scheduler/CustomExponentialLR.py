from typing import List

import torch


__all__ = ["CustomExponentialLR"]


class CustomExponentialLR(torch.optim.lr_scheduler.ExponentialLR):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 gamma: float,
                 last_epoch=-1,
                 verbose="deprecated"
                 ) -> None:
        if last_epoch == -1:
            for group in optimizer.param_groups:
                initial_lr = group["lr"]
                if isinstance(initial_lr, torch.Tensor):
                    initial_lr = initial_lr.clone()
                group.setdefault("lr", initial_lr)
        else:
            for i, group in enumerate(optimizer.param_groups):
                if "lr" not in group:
                    raise KeyError(
                        "param 'lr' is not specified "
                        f"in param_groups[{i}] when resuming an optimizer"
                    )

        self.base_lrs: List[float] = [group["lr"] for group in optimizer.param_groups]
        super().__init__(optimizer, gamma, last_epoch, verbose)

    def get_lr(self):
        """Compute the learning rate of each parameter group."""
        if self.last_epoch == 0:
            return self.base_lrs

        self.base_lrs = [min(1., base_lr * self.gamma ** self.last_epoch) for base_lr in self.base_lrs]
        return self.base_lrs
