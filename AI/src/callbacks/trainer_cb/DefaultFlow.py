import os
import copy
import pathlib
from dataclasses import replace
from typing import Tuple, Dict, Any

import torch

from ...runner import Trainer
from .BaseCallback import BaseCallback
from ...utils import ModelArchInspector, get_amp_cfg


__all__ = ["DefaultFlow"]


class DefaultFlow(BaseCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """
    def on_init_end(self, instance: Trainer) -> None:
        """
        Resume from specified checkpoint. By default, checkpoint contains
            1/ Model's weight if save_weights_only=True, otherwise torch.nn.Module model
            2/ Optimizer state dict
            3/ Scheduler state dict (if have)
            4/ Training config (if have)
            5/ Best metric according to Checkpoint callback
            6/ Trained epochs & steps
        """
        epoch: int = 1
        epochs: int = instance.config.Global.get("epochs", 1)

        step: int = 0
        steps: int = epochs * instance.train_dataloader.__len__()

        # Calculate eval_steps (if have)
        eval_strategy: str = instance.config.Global.get("eval_strategy", "no")
        assert eval_strategy in ("no", "step", "epoch")

        eval_steps: None = None
        if eval_strategy == "step":
            eval_steps: int = instance.config.Global.get("eval_steps", instance.val_dataloader.__len__())
        elif eval_strategy == "epoch":
            eval_steps: int = instance.val_dataloader.__len__()

        # Resume training
        if instance.config.Global.get("resume", False):
            # point directly to checkpoint
            resume_ckpt: str = str(pathlib.Path(instance.config.Global.get("resume_ckpt", "")))
            assert os.path.isfile(resume_ckpt), FileNotFoundError

            ckpt: Dict[str, Any] | torch.nn.Module = torch.load(f=resume_ckpt, map_location="cpu")

            epoch += ckpt["epoch"]
            step += ckpt["step"] + 1
            steps = step + epochs * instance.train_dataloader.__len__()

            monitor = ckpt["monitor"]

            # Load model's weights
            if isinstance(ckpt["model"], dict):
                instance.model.load_state_dict(ckpt["model"])
            else:
                instance.model.load_state_dict(ckpt["model"].state_dict())

            # Load model's weights to optim & scheduler
            instance.optim.load_state_dict(ckpt["optim"])
            for state in instance.optim.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(instance.config.Global.get("device", "cpu"))

            instance.scheduler.load_state_dict(ckpt["scheduler"])
            del ckpt
        else:
            monitor = None

        # Update state
        instance.state = replace(instance.state, **{
            "step": step,
            "epoch": epoch,
            "steps": steps,
            "epochs": epochs,
            "eval_steps": eval_steps,
            "eval_strategy": eval_strategy,
            "monitor": monitor
        })

        # Inspect model architecture
        inspect_model_arch: bool = instance.config.Global.get("inspect_model_arch", False)
        dummy_shape: None | Tuple[int, ...] = instance.config.Global.get("dummy_input_shape", None)

        if inspect_model_arch and dummy_shape is not None:
            amp_cfg, _ = get_amp_cfg(instance.config)
            with torch.amp.autocast(**amp_cfg):
                model_arch = ModelArchInspector(
                    copy.deepcopy(instance.model),
                    instance.config.Global.dummy_input_shape,
                    depth=instance.config.Global.get("inspect_depth", 3),
                    verbose=0,
                    mode="train",
                    device=instance.config.Global.get("device", "cpu")
                )
            instance.config["Model_arch"] = str(model_arch())

    def on_train_begin(self, instance: Trainer) -> None:
        instance.state.phase = "train"

    def on_train_end(self, instance: Trainer) -> None:
        pass

    def on_train_epoch_begin(self, instance: Trainer) -> None:
        instance.state.batch_output = None
        instance.control.should_epoch_stop = False

    def on_train_epoch_end(self, instance: Trainer) -> None:
        instance.state.epoch += 1

        if instance.config.Global.get("eval_strategy", "no") == "epoch":
            instance.state.phase = "train"

    def on_val_epoch_begin(self, instance: Trainer):
        instance.state.batch_output = None
        instance.control.should_evaluate = False

    def on_val_epoch_end(self, instance: Trainer):
        instance.state.step += 1
        instance.state.phase = "train"

        # Update monitor and will be overridden if EarlyStopping callback is used
        if instance.state.batch_output.phase == "val":
            instance.state.monitor = (instance.state.monitor[0], instance.state.batch_output.loss)

    def on_step_begin(self, instance: Trainer) -> None:
        instance.control.should_log = False
        instance.control.should_save = False
        instance.control.should_evaluate = False

    def on_step_end(self, instance: Trainer) -> None:
        # Currently, log at the end of train/ val phase
        if (instance.state.batch_output.step + 1) % len(getattr(instance, f"{instance.state.phase}_dataloader")) == 0:
            instance.control.should_log = True

        # Evaluate
        if (
                instance.state.batch_output.phase == "train" and
                instance.state.eval_strategy == "step" and
                instance.state.step > 0 and
                (instance.state.step + 1) % instance.state.eval_steps == 0
        ) or (
                instance.state.batch_output.phase == "train" and
                instance.state.eval_strategy == "epoch" and
                instance.state.step + 1 == instance.state.epoch * len(instance.train_dataloader)
        ):
            instance.state.phase = "val"
            instance.control.should_evaluate = True

        # Checkpointing logic is relegated to ModelCheckpoint callback

        # End training
        if instance.state.step >= instance.state.steps:
            instance.control.should_training_stop = True

        # Update step
        if instance.state.phase == "train":
            instance.state.step += 1
