from typing import Dict, Any, Callable, Union

import torch
from tqdm import tqdm
from torch import GradScaler, Tensor

from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from AI.src.tools import Trainer
from AI.src.losses import LossWrapper
from AI.src.data.model import BatchOutput


__all__ = ["FORWARD_STRATEGIES"]


def v1(instance: Union[Trainer],
       phase: str,
       model: Module,
       dataloader: DataLoader,
       amp_cfg: Dict[str, Any],
       epochs: int,
       cur_epoch: int,
       device: str,
       grad_ctx_manager,
       loss: LossWrapper = None,
       optim: Optimizer = None,
       scheduler: LRScheduler = None,
       grad_scaler: GradScaler = None,
       ) -> None:
    """
    This phase use for train/ val single MIL VAD problem

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
    lr: None | float = None
    cur_step: int = (cur_epoch - 1) * len(dataloader)

    for i, (inps, labels) in tqdm(enumerate(dataloader), initial=cur_step, total=len(dataloader) * epochs, desc=f"Forward v2, Phase: {phase}, Epoch: {cur_epoch}"):
        inps: Tensor

        if phase == "train" and optim is not None:
            optim.zero_grad()

        with grad_ctx_manager, torch.amp.autocast(**amp_cfg):
            anomaly, normal = torch.chunk(inps, 2, 1)

            anomaly_preds: Tensor = model(anomaly.to(device)).preds  # (B, S)
            normal_preds: Tensor = model(normal.to(device)).preds  # (B, S)

            batch_loss: Tensor = loss.compute_batch_loss([anomaly_preds, normal_preds])

        # Exits the context manager before backward
        if phase == "train":
            if grad_scaler is not None:
                grad_scaler.scale(batch_loss).backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm) add later on
                grad_scaler.step(optim)
                grad_scaler.update()
            else:
                batch_loss.backward()
                optim.step()

            if scheduler is not None:
                scheduler.step(cur_step)

            lr: float = optim.param_groups[-1]["lr"] if instance.scheduler is None else scheduler.get_last_lr()[-1]

        # Per step logging
        batch_output: Dict[str, Any] = {
            "phase": phase,
            "cur_step": cur_step,
            "lr": lr,
            "loss": batch_loss.item(),
        }
        instance.batch_output = BatchOutput(**batch_output)

        # Update step
        cur_step += 1

        if i == 3:
            break
        instance.run_callbacks(f"on_{phase}_batch_end")


def v2(T_max: int = 50,
       frame_overlap_ratio: float = 0.5
       ) -> None:
    # elif phase in ("val", "test"):
    #     cur_step: int = 0 + cur_epoch * len(dataloader)
    #
    #     for i, (inps, labels) in tqdm(enumerate(dataloader), initial=cur_step, total=len(dataloader) * epochs, desc=f"Forward v2, Phase: {phase}, Epoch: {cur_epoch + 1}"):
    #         inps: Tensor = inps.squeeze()  # (B,C,T,H,W) -> (C,T,H,W)
    #         labels: Tensor = labels.squeeze()  # (T,)
    #
    #         total_frames: int = inps.shape[1]
    #         preds: Tensor = torch.zeros_like(labels, dtype=torch.float16)
    #
    #         with grad_ctx_manager, torch.amp.autocast(**amp_cfg):
    #             for j in range(total_frames):
    #                 if j < T_max:
    #                     pad = (0, 0, 0, 0, T_max-j, 0)
    #                     model_inps = inps[:, :j, ...]
    #                     model_inps = torch.nn.ZeroPad3d(pad)(model_inps)
    #                 else:
    #                     model_inps = inps[:, j-T_max:j, ...]
    #
    #                 model_inps = model_inps.unsqueeze(0)
    #                 pred: Tensor = model(model_inps.to(device)).preds
    #                 preds[j] = pred
    #
    #         metrics.update(preds, labels)
    #
    #         # Per step logging
    #         batch_output: Dict[str, Any] = {
    #             "phase": phase,
    #             "cur_step": cur_step,
    #             "loss": batch_loss.item(),
    #         }
    #
    #         lr: float = optim.param_groups[-1]["lr"] if instance.scheduler is None else scheduler.get_last_lr()[-1]
    #         batch_output["lr"] = lr
    #
    #         # Update step
    #         cur_step += 1
    #
    #         if i == 0:
    #             break
    #
    #         instance.batch_output = BatchOutput(**batch_output)
    #         instance.run_callbacks("on_train_batch_end")
    #
    #         if i == 0:
    #             break
    #
    #     metrics.compute()
    #     a=metrics.get_result()
    # return None
    raise NotImplementedError


FORWARD_STRATEGIES: Dict[str, Callable] = {
    "v1": v1,
    "v2": v2,

}
