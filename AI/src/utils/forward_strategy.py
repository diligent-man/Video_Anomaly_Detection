from typing import Dict, Any, Callable, Union, List


import torch

from torch import GradScaler, Tensor
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.autograd.grad_mode import set_grad_enabled, inference_mode


import AI.src.utils.BatchForwarder as BatchForwarder  # Circular dependency

from .DotDict import DotDict
from ..runner import Trainer
from ..losses import LossWrapper
from ..data.model import BatchOutput
from ..modeling.architectures import BaseModelOutput

from ..utils.runner_utils.trainer import find_initial_total


__all__ = ["FORWARD_STRATEGIES"]


def v1(instance: Union[Trainer],
       grad_ctx: set_grad_enabled | inference_mode,
       dataloader: DataLoader,
       amp_cfg: Dict[str, Any],
       grad_scaler: GradScaler = None,
       loss: LossWrapper = None,
       optim: Optimizer = None,
       scheduler: LRScheduler = None
       ) -> None:
    phase = instance.state.phase
    device: str = instance.config.Global.get("device", "cpu")
    initial, _ = find_initial_total(instance, dataloader)

    instance.callback(f"on_{phase}_epoch_begin")
    for step, (inps, _) in enumerate(dataloader):
        inps: Tensor
        lr: None | float = None
        batch_loss: None | torch.Tensor = None

        if phase == "train" and optim is not None:
            # safer than optimizer.zero_grad() in case of num of optimizer > 1
            instance.model.zero_grad()

        instance.callback("on_step_begin")
        with grad_ctx, torch.amp.autocast(**amp_cfg):
            anomaly, normal = torch.chunk(inps, 2, 1)

            anomaly_preds: Tensor = instance.model(anomaly.to(device)).preds  # (B, S)
            normal_preds: Tensor = instance.model(normal.to(device)).preds  # (B, S)

            if loss is not None:
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
                scheduler.step(instance.state.step)

            lr = optim.param_groups[-1]["lr"] if instance.scheduler is None else scheduler.get_last_lr()[-1]

        # Per step logging
        batch_output: Dict[str, Any] = {
            "phase": phase,
            "epoch": instance.state.epoch,
            "step": initial+step,
            "lr": lr,
            "loss": batch_loss.item() if batch_loss is not None else batch_loss,
        }

        instance.state.batch_output = BatchOutput(**batch_output)
        instance.callback(f"on_step_end")

        if instance.control.should_evaluate:
            BatchForwarder.BatchForwarder(
                instance.config.Data[phase].forward_strategy,
                instance,
                **{
                    "overridden_args": instance.config.Data[phase].get("overridden_args", DotDict({})).get_dict()
                }
            )()
    instance.callback(f"on_{phase}_epoch_end")


def v2(instance: Union[Trainer],
       grad_ctx: set_grad_enabled | inference_mode,
       dataloader: DataLoader,
       amp_cfg: Dict[str, Any],
       grad_scaler: GradScaler = None,
       loss: LossWrapper = None,
       optim: Optimizer = None,
       scheduler: LRScheduler = None
       ) -> None:
    phase = instance.state.phase
    device: str = instance.config.Global.get("device", "cpu")
    initial, _ = find_initial_total(instance, dataloader)

    instance.callback(f"on_{phase}_epoch_begin")
    for step, (inps, _) in enumerate(dataloader):
        inps: Tensor
        lr: None | float = None
        batch_loss: None | torch.Tensor = None

        if phase == "train" and optim is not None:
            # safer than optimizer.zero_grad() in case of num of optimizer > 1
            instance.model.zero_grad()

        instance.callback("on_step_begin")
        with grad_ctx, torch.amp.autocast(**amp_cfg):
            anomaly, normal = torch.chunk(inps, 2, 1)
            student_outs, teach_outs = instance.model(anomaly, normal, device)

            if loss is not None:
                batch_loss: Tensor = loss.compute_batch_loss(student_outs, teach_outs)

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
                scheduler.step(instance.state.step)

            lr = optim.param_groups[-1]["lr"] if instance.scheduler is None else scheduler.get_last_lr()[-1]

        # Per step logging
        batch_output: Dict[str, Any] = {
            "phase": phase,
            "epoch": instance.state.epoch,
            "step": initial+step,
            "lr": lr,
            "loss": batch_loss.item() if batch_loss is not None else batch_loss,
        }

        instance.state.batch_output = BatchOutput(**batch_output)
        instance.callback(f"on_step_end")

        if instance.control.should_evaluate:
            BatchForwarder.BatchForwarder(
                instance.config.Data[phase].forward_strategy,
                instance,
                **{
                    "overridden_args": instance.config.Data[phase].get("overridden_args", DotDict({})).get_dict()
                }
            )()
    instance.callback(f"on_{phase}_epoch_end")


def v3(T_max: int = 50,
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
