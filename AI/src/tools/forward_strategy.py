from typing import Dict, Any, Callable

import torch
from tqdm import tqdm
from torch import GradScaler, Tensor

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader


from ..losses import LossWrapper
from ..metrics import MetricWrapper
from ..data.model import BatchOutput


__all__ = ["FORWARD_STRATEGIES"]


def v1(phase: str,
       model: Module,
       optim: Optimizer,
       scheduler: LRScheduler,
       dataloader: DataLoader,
       metrics: MetricWrapper,
       amp_cfg: Dict[str, Any],
       num_classes: int,
       grad_scaler: GradScaler = None,
       ) -> BatchOutput:
    """
    for i, (inps, targets) in enumerate(DataLoader):
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
       ) -> BatchOutput:
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
    """
    cur_step: int = 0 + cur_epoch * len(dataloader)

    for i, (inps, _) in tqdm(enumerate(dataloader), initial=cur_step, total=len(dataloader) * epochs, desc=f"Foward v2, Phase: {phase}, Epoch: {cur_epoch+1}"):
        inps: Tensor

        with grad_ctx_manager, torch.amp.autocast(**amp_cfg):
            anomaly, normal = torch.chunk(inps, 2, 1)

            # (B, S)
            anomaly_preds: Tensor = model(anomaly.to(device)).preds
            normal_preds: Tensor = model(normal.to(device)).preds

            loss = loss.compute_batch_loss([anomaly_preds, normal_preds], )
            # model_output = model(imgs, labels, model, optimizer, metrics, loss, phase, device)
            # total_loss += batch_loss.item()  # Accumulate minibatch into total loss
        break

# def _forward(imgs: Tensor, labels: Tensor, num_classes: int,
#              model: Module, optimizer: Optimizer,
#              metrics: MetricWrapper,
#              phase: str, device: str
#              ) -> FloatTensor:
#     """
#     Computation task in forward pass:
#     1. Pass through model
#     2. Compute batch loss
#     3. Update metrics
#
#     Return:
#         batch_loss
#     """
#     def _activate(pred_labels: torch.Tensor) -> torch.Tensor:
#         if pred_labels.shape[1] == 1:
#             # Binary class
#             return torch.nn.functional.sigmoid(pred_labels).squeeze(dim=1)
#         else:
#             # Multiclass
#             return torch.nn.functional.softmax(pred_labels, dim=1)
#
#     imgs = imgs.to(device, non_blocking=True)
#
#     labels = labels.type(torch.FloatTensor) if num_classes == 1 else labels.type(torch.LongTensor)
#     labels = labels.to(device, non_blocking=True)
#
#     # reset gradients prior to forward pass
#     optimizer.zero_grad()
#
#     with torch.set_grad_enabled(phase == "train"):
#         # forward pass
#         pred_labels = model(imgs)
#         pred_labels = list(map(_activate, pred_labels)) if isinstance(pred_labels, Tuple) else _activate(pred_labels)
#
#         # Compute loss
#         batch_loss: FloatTensor = loss.compute_batch_loss(pred_labels, labels)
#
#         # Get pred_labels from main output
#         if isinstance(pred_labels, List): pred_labels = pred_labels[0]
#
#         # Update metrics only if eval phase or metric_in_train == True
#         if metrics: metrics.update(pred_labels, labels)
#     return batch_loss


FORWARD_STRATEGIES: Dict[str, Callable] = {
    "v1": v1,
    "v2": v2
}

