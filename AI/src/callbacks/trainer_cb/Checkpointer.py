import os
import glob
from typing import Dict, Any, Tuple, List

import torch

from ...runner import Trainer
from .BaseCallback import BaseCallback


__all__ = ["Checkpointer"]


class Checkpointer(BaseCallback):
    """
    This callback idea is adopted from Keras src
    """
    __support_monitor: Tuple[str] = "val_loss"  # currently
    __support_mode: Tuple[str] = "min"  # currently

    __save_dir: str
    __monitor: str
    __save_freq: str | int
    __save_best_only: bool
    __save_weights_only: bool
    __save_total_limit: int
    __include_config: bool
    __best: float
    __verbose: bool

    def __init__(self,
                 save_dir: str,
                 mode: str = "min",
                 monitor: str = "val_loss",
                 save_freq: str | int = "epoch",
                 save_best_only: bool = False,
                 save_weights_only: bool = False,
                 save_total_limit: int = 5,
                 include_config: bool = False,
                 initial_value_threshold: float = None,
                 verbose: bool = True
                 ) -> None:
        assert mode in self.__support_mode, ValueError(f"Currently support {self.__support_mode}. Get {mode}")
        assert monitor in self.__support_monitor, ValueError(
            f"Currently support {self.__support_monitor}. Get {monitor}")

        if isinstance(save_freq, str):
            assert save_freq == "epoch", ValueError(f"'epoch' or int for step saving. Get {save_freq}, type: {type(save_freq)}")
        else:
            assert isinstance(save_freq, int), ValueError(f"'epoch' or int for step saving. Get {save_freq}, type: {type(save_freq)}")
        super(Checkpointer, self).__init__()

        self.__save_dir = save_dir
        self.__monitor = monitor
        self.__save_freq = save_freq
        self.__save_best_only = save_best_only
        self.__save_weights_only = save_weights_only
        self.__save_total_limit = save_total_limit
        self.__include_config = include_config
        self.__best = initial_value_threshold
        self.__verbose = verbose

        self.__last_step_seen = 0
        self.__steps_seen_since_last_saving = 0

        if mode == "min":
            self.__monitor_op = torch.less
            if self.__best is None:
                self.__best = torch.inf

    def __repr__(self) -> str:
        return f"""Save dir: {self.__save_dir}
Monitor: {self.__monitor}
Save freq: {self.__save_freq}
Best: {self.__best}"""

    def on_train_epoch_end(self, instance: Trainer) -> None:
        if (
                self.__save_freq == "epoch" and
                instance.state.batch_output.phase == "val" and
                (instance.state.batch_output.step + 1) % len(instance.val_dataloader) == 0
        ):
            save_obj: Dict[str, Any] = self._make_save_obj(instance)
            self._save_model(save_obj, instance.state.epoch-1, instance.state.step)

    def on_step_end(self, instance: Trainer):
        if (
                self._should_save_on_step(instance.state.step) and
                instance.state.batch_output.phase == "val" and
                (instance.state.batch_output.step + 1) % len(instance.val_dataloader) == 0
        ):
            save_obj: Dict[str, Any] = self._make_save_obj(instance)
            self._save_model(save_obj, instance.state.epoch, instance.state.step)

    def _should_save_on_step(self, cur_step: int) -> bool:
        """Handles batch-level saving logic, supports steps_per_execution."""
        if self.__save_freq == "epoch":
            return False
        else:
            if (cur_step + 1) % self.__save_freq == 0:
                return True
            else:
                return False

    def _make_save_obj(self, instance: Trainer) -> Dict[str, Any]:
        save_obj: Dict[str, Any] = {
            "model": instance.model.state_dict() if self.__save_weights_only else instance.model,
            "optim": instance.optim.state_dict(),
            "scheduler": instance.scheduler.state_dict() if instance.scheduler is not None else None,
            "epoch": instance.state.epoch-1,
            "step": instance.state.step,
            "monitor": (self.__monitor, instance.state.batch_output.loss)
        }

        if self.__include_config:
            save_obj["config"] = instance.config

        torch.serialization.add_safe_globals([Dict[str, Any]])
        return save_obj

    def _save_model(self, obj: Dict[str, Any], epoch: int, step: int) -> None:
        # Best ckpt is always saved regardless of save_freq
        if self.__monitor_op(torch.Tensor([obj["monitor"][1]]), self.__best):
            # remove previous best epoch
            for name in os.listdir(self.__save_dir):
                if name.startswith("best"):
                    filepath = os.path.join(self.__save_dir, name)
                    os.remove(filepath)
                    break

            torch.save(obj, os.path.join(self.__save_dir, f"best_epoch{epoch}_step{step}.pt"))
            if self.__verbose:
                cur_val = obj["monitor"][1]
                print(
                    f"\nEpoch {epoch}: {self.__monitor} improved from {self.__best:.5f} to {cur_val:.5f}")

            self.__best = obj["monitor"][1]
        else:
            if self.__verbose:
                print(f"Epoch {epoch}: {self.__monitor} did not improve from {self.__best:.5f}")

        # Save by freq
        if not self.__save_best_only:
            torch.save(obj, os.path.join(self.__save_dir, f"epoch{epoch}_step{step}.pt"))
            if self.__verbose:
                print(f"Ckpt is saved at epoch {epoch}, step {step}, cur: {obj['monitor'][1]}, best: {self.__best}")

            save_ckpt_lst: List[str] = [f for f in glob.glob(f"{self.__save_dir}/epoch*.pt")]
            save_ckpt_lst: List[str] = sorted(save_ckpt_lst, key=os.path.getctime, reverse=True)

            if len(save_ckpt_lst) > self.__save_total_limit:
                for i in range(self.__save_total_limit, len(save_ckpt_lst)):
                    os.remove(save_ckpt_lst[i])
