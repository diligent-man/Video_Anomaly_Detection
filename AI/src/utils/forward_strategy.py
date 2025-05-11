import gc
from typing import Dict, Any, Callable, Tuple, List


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
from ..data.model import BatchOutput
from ..runner import Trainer, Tester

from ..utils import VideoCache
from ..utils.inference_ops import dispatch_infer
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
    """
    Teacher training
    """
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
    """
    Student training
    """
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
            student_outs, teacher_outs = instance.model(anomaly.to(device), normal.to(device))

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
       # metric: MetricWrapper,
       T_max: int = 30,
       overlap_ratio: float = 0.5,
       ) -> None:
    """
    Test VAD model
    """
    device: str = instance.config.Global.get("device", "cpu")
    batch_thres = instance.config.Global.get("batch_thres", None)
    batch_worker = instance.config.Global.get("batch_worker", None)

    phase: str = instance.state.phase
    video_cache = VideoCache(batch_thres, batch_worker)

    for idx, inp, label in tqdm(dataloader, total=len(dataloader), desc=f"Forward v3, Phase: {phase}"):
        idx: torch.Tensor
        inp: Tuple[str]

        idx: int = idx.item()
        inp: str = inp[0]

        video_cache.cache(label.squeeze(0).shape[0], ["inp", "label", "idx"], [inp, label, idx])

        cache: Dict[str, Any] | None = video_cache.get_cache()

        if cache is not None:
            print("Batch:", cache["batch_worker"])
            result: Tuple[List[float], List[int]] = dispatch_infer(
                cache, instance.model, device,
                T_max, overlap_ratio, True,
                amp_cfg, grad_ctx
            )

            for i in range(len(result)):
                instance.state.step_info = f"{result[i][0]}; {result[i][1]}; {cache['idx'][i]}"
                instance.callback(f"on_step_end")

            gc.collect()
            torch.cuda.empty_cache()

    # Remaining
    print("Run leftovers")
    for cache in video_cache.get_remains():
        print("Batch:", cache["batch_worker"])
        result: Tuple[List[float], List[int]] = dispatch_infer(
                cache, instance.model, device,
                T_max, overlap_ratio, True,
                amp_cfg, grad_ctx
            )

        for i in range(len(result)):
            instance.state.step_info = f"{result[i][0]},{result[i][1]},{cache['idx'][i]}"
            instance.callback(f"on_step_end")

        gc.collect()
        torch.cuda.empty_cache()
    return None


FORWARD_STRATEGIES: Dict[str, Callable] = {
    "v1": v1,
    "v2": v2,
    "v3": v3
}
