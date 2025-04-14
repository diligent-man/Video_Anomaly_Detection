from typing import Dict, Any, Callable, List


import torch

from tqdm import tqdm
from torch import GradScaler, Tensor
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.autograd.grad_mode import set_grad_enabled, inference_mode


import AI.src.utils.BatchForwarder as BatchForwarder  # Circular dependency

from .DotDict import DotDict
from ..losses import LossWrapper
from ..metrics import MetricWrapper
from ..data.model import BatchOutput
from ..runner import Trainer, Tester

from ..utils.runner_utils.trainer import find_initial_total


__all__ = ["FORWARD_STRATEGIES"]


def v1(instance: Trainer,
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


def v2(instance: Trainer,
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
            student_outs, teacher_outs = instance.model(anomaly, normal, device)

            if loss is not None:
                batch_loss: Tensor = loss.compute_batch_loss(student_outs, teacher_outs)

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


def v3(instance: Tester,
       grad_ctx: set_grad_enabled | inference_mode,
       dataloader: DataLoader,
       amp_cfg: Dict[str, Any],
       metric: MetricWrapper,
       grad_scaler: GradScaler = None,
       T_max: int = 30,
       overlap_ratio: float = 0.5,
       ) -> None:
    device: str = instance.config.Global.get("device", "cpu")

    phase: str = instance.state.phase
    total_labels: None | Tensor = None
    total_preds: None | Tensor = None
    for i, (inp, label) in tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Forward v3, Phase: {phase}"):
        inp: Tensor = inp.squeeze()  # (B,T,C,H,W) -> (T,C,H,W)
        inp = inp.permute(1, 0, 2, 3).unsqueeze(0).unsqueeze(0)  # (T,C,H,W) -> (1,1,C,T,H,W)

        label: Tensor = label.squeeze()  # (T,)

        total_frames: int = label.shape[0]
        total_labels = label if total_labels is None else torch.cat((total_labels, label), 0)

        cum_frames: int = 0
        step_preds: None | Tensor = None
        preds: Tensor = torch.zeros_like(label, dtype=amp_cfg["dtype"])
        with grad_ctx, torch.amp.autocast(**amp_cfg):
            for j in range(total_frames):
                if j < T_max or cum_frames < T_max:
                    cum_frames += 1
                else:
                    # step when accumulate sufficient frames
                    step_preds: Tensor = instance.model(inp[:, :, :, j-cum_frames:j, ...].to(device)).preds  # (B, S)
                    cum_frames = int(T_max * overlap_ratio) + 1

                # Last step
                if j == total_frames-1:
                    step_preds: Tensor = instance.model(inp[:, :, :, j-cum_frames:j, ...].to(device)).preds  # (B, S)

                if step_preds is not None:
                    step_preds = step_preds.squeeze(0).to("cpu")
                    # First half
                    if preds[j-T_max: j-(T_max//2)].equal(torch.zeros_like(preds[j-T_max: j-(T_max//2)], dtype=preds.dtype)):
                        preds[j - T_max: j - (T_max // 2)] += step_preds
                    else:
                        preds[j - T_max: j - (T_max // 2)] = (preds[j - T_max: j - (T_max // 2)] + step_preds) / 2

                    # Second half
                    if j == total_frames - 1:
                        preds[j - (T_max // 2): ] += step_preds
                    else:
                        preds[j - (T_max // 2): j] += step_preds

                    # Reset
                    step_preds = None
            total_preds = preds if total_preds is None else torch.cat((total_preds, preds), 0)
        instance.state.preds = preds
        instance.callback(f"on_step_end")
        if i == 1:
            break
    metric.update(total_preds, total_labels)
    metric.compute()
    instance.state.metric_result = metric.get_result(return_dict=True)
    return None


FORWARD_STRATEGIES: Dict[str, Callable] = {
    "v1": v1,
    "v2": v2,
    "v3": v3
}
