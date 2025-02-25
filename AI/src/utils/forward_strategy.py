from typing import Dict, Any, Callable, Union, List

import torch
from tqdm import tqdm
from torch import GradScaler, Tensor

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from AI.src.tools import Trainer
from AI.src.losses import LossWrapper
from AI.src.metrics import MetricWrapper
from AI.src.data.model import BatchOutput


__all__ = ["FORWARD_STRATEGIES"]


def v1(instance: Union[Trainer],
       phase: str,
       epochs: int,
       cur_epoch: int,
       grad_ctx_manager,
       model: Module,
       dataloader: DataLoader,
       loss: LossWrapper,
       metrics: MetricWrapper,
       amp_cfg: Dict[str, Any],
       optim: Optimizer = None,
       scheduler: LRScheduler = None,
       grad_scaler: GradScaler = None,
       device: str = "cpu"
       ) -> None:
    """
    for i, (inps, targets) in enumerate(DataLoader):
        forward -> batch loss
        backward + step optim (phase == "train")
        step scheduler (phase == "val")

    for i, (inps, targets) in enumerate(DataLoader):
        forward model
        compute batch loss

        if phase == "train:
            backward
            step optim

            if scheduler is not None:
                step scheduler

        compute metrics (if have)
    """
    cur_step: int = 0 + cur_epoch * len(dataloader)

    for i, (inps, labels) in tqdm(enumerate(dataloader), initial=cur_step, total=len(dataloader) * epochs, desc=f"Foward v2, Phase: {phase}, Epoch: {cur_epoch+1}"):
        inps: Tensor

        if phase == "train":
            optim.zero_grad()

        with grad_ctx_manager, torch.amp.autocast(**amp_cfg):
            anomaly, normal = torch.chunk(inps, 2, 1)

            anomaly_preds: Tensor = model(anomaly.to(device)).preds  # (B, S)
            normal_preds: Tensor = model(normal.to(device)).preds  # (B, S)

            batch_loss: Tensor = loss.compute_batch_loss([anomaly_preds, normal_preds])

        # Exits the context manager before backward and compute metrics
        if phase == "train":
            batch_loss.backward()
            optim.step()

            if scheduler is not None:
                scheduler.step(cur_step)

            if metrics.in_train:
                metrics.update(torch.hstack((anomaly_preds, normal_preds)), labels)
        else:
            metrics.update(torch.hstack((anomaly_preds, normal_preds)), labels)

        # Per step logging
        batch_output: Dict[str, Any] = {
            "phase": phase,
            "cur_step": cur_step,
            "loss": batch_loss.item(),
        }

        if phase == "train":
            lr: float = optim.param_groups[-1]["lr"] if instance.scheduler is None else scheduler.get_last_lr()[-1]
            batch_output["lr"] = lr

        # Update step
        cur_step += 1

        if i == 3:
            break
        instance.batch_output = BatchOutput(**batch_output)
        instance.run_callbacks("on_train_batch_end")
    return None


FORWARD_STRATEGIES: Dict[str, Callable] = {
    "v1": v1
}
