from typing import Dict, Any, Callable, Union

import torch
from torch import GradScaler, Tensor
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.autograd.grad_mode import set_grad_enabled, inference_mode

from .DotDict import DotDict
from ..runner import Trainer
from ..losses import LossWrapper
from ..data.model import BatchOutput
from ..utils.runner_utils.trainer import find_initial_total
import AI.src.utils.BatchForwarder as BatchForwarder  # Circular dependency


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
    phase = instance.state.phase
    device: str = instance.config.Global.get("device", "cpu")
    initial, _ = find_initial_total(instance, dataloader)

    instance.callback(f"on_{phase}_epoch_begin")
    for step, (inps, labels) in enumerate(dataloader):
        inps: Tensor
        lr: None | float = None
        batch_loss: None | torch.Tensor = None

        if phase == "train" and optim is not None:
            instance.model.zero_grad()  # safer than optimizer.zero_grad() in case of num of optimizer > 1

        from ..utils.saving import save_video
        instance.callback("on_step_begin")
        with grad_ctx, torch.amp.autocast(**amp_cfg):
            anomaly, normal = torch.chunk(inps, 2, 1)

            # if instance.state.phase == "train":
            #     _ = [
            #         save_video(
            #             255 * anomaly[i, ...].permute(0, 2, 3, 4, 1).reshape(-1, 224, 224, 3),
            #             "/home/trong/Downloads/tmp",
            #             15,
            #             f"out_video_phase_{instance.state.phase}_epoch_{instance.state.epoch}_step_{instance.state.step}_{i}"
            #         )
            #         for i in range(anomaly.shape[0])
            #     ]

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
