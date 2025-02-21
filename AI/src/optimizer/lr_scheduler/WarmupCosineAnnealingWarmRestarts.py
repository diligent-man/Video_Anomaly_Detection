import math

import overrides
import torch


__all__ = ["WarmupCosineAnnealingWarmRestarts"]


class WarmupCosineAnnealingWarmRestarts(torch.optim.lr_scheduler.CosineAnnealingWarmRestarts):
    def __init__(self,
                 optimizer: torch.optim.Optimizer,
                 t_0: int,
                 t_mult: int = 1,
                 num_warmup_steps: int = 1,
                 eta_min: float = 0.0,
                 last_epoch: int = -1,
                 verbose: str = "deprecated",
                 ) -> None:
        self.num_warmup_steps: int = num_warmup_steps
        super(WarmupCosineAnnealingWarmRestarts, self).__init__(optimizer, t_0, t_mult, eta_min, last_epoch, verbose)

    @overrides.override
    def get_lr(self):
        """Compute the initial learning rate."""
        # Warmup phase
        if self.last_epoch <= self.num_warmup_steps:
            return [
                base_lr * self.last_epoch * 1.0 / self.num_warmup_steps
                for base_lr in self.base_lrs
            ]
        # Annealing phase
        else:
            return [
                self.eta_min
                + (base_lr - self.eta_min)
                * (1 + math.cos(math.pi * self.T_cur / self.T_i))
                / 2
                for base_lr in self.base_lrs
            ]
