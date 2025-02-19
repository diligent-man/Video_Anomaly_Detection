from typing import Dict, Any, Callable


import torch
from tqdm import tqdm


from ..data.model import BatchOutput
from ..metrics import MetricManager


__all__ = ["FORWARD_STRATEGIES"]


def v1(phase: str,
       model: torch.nn.Module,
       optim: torch.optim.Optimizer,
       scheduler: torch.optim.lr_scheduler.LRScheduler,
       dataloader: torch.utils.data.DataLoader,
       metrics: MetricManager,
       amp_cfg: Dict[str, Any],
       num_classes: int,
       grad_scaler: torch.GradScaler = None,
       ) -> BatchOutput:
    """
    for i, (inps, targets) in enumerate(dataloader):
        forward model
        compute batch loss

        if phase == "train:
            backward
            step optim

    if phase == "val" and scheduler is not None:
        step scheduler

    Implement later on
    """
    raise NotImplementedError


def v2(phase: str,
       epochs: int,
       cur_epoch: int,
       ctx_manager,
       model: torch.nn.Module,
       dataloader: torch.utils.data.DataLoader,
       metrics: MetricManager,
       amp_cfg: Dict[str, Any],
       optim: torch.optim.Optimizer = None,
       scheduler: torch.optim.lr_scheduler.LRScheduler = None,
       grad_scaler: torch.GradScaler = None,
       ) -> BatchOutput:
    """
    for i, (inps, targets) in enumerate(dataloader):
        forward -> batch loss
        backward + step optim (phase == "train")
        step scheduler (phase == "val")

    for i, (inps, targets) in enumerate(dataloader):
        forward model
        compute batch loss

        if phase == "train:
            backward
            step optim

            if scheduler is not None:
                step scheduler
    """
    cur_step: int = 0 + cur_epoch * len(dataloader)

    for i, j in tqdm(enumerate(dataloader), initial=cur_step, total=len(dataloader) * epochs):
        print(i)


FORWARD_STRATEGIES: Dict[str, Callable] = {
    "v1": v1,
    "v2": v2
}
