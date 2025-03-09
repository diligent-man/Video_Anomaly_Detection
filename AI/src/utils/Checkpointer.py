# TODO: Add remove previous checkpoints based on certain conditions
# TODO: Review save checkpoint manner
import os
from typing import Tuple, Dict, Any

import torch


__all__ = ["Checkpointer"]


class Checkpointer(object):
    __support_monitor: Tuple[str] = "val_loss"
    __support_mode: Tuple[str] = "min"

    __save_dir: str
    __monitor: str
    __save_freq: str | int
    __save_best_only: bool
    __save_weights_only: bool
    __save_total_limit: int
    __include_config: bool
    __best: float
    __verbose: bool

    __batches_seen_since_last_saving: int
    __last_batch_seen: int
    cur_epoch: int

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
                 verbose: bool = True,
                 ) -> None:
        super(Checkpointer, self).__init__()
        assert mode in self.__support_mode, ValueError(f"Currently support {self.__support_mode}. Get {mode}")
        assert monitor in self.__support_monitor, ValueError(f"Currently support {self.__support_monitor}. Get {monitor}")
        assert isinstance(save_freq, (str, int)), ValueError(f"'epoch' or int for step saving. Get {save_freq}")

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

    # def on_train_batch_end(self, batch, logs=None):
    #     if self._should_save_on_batch(batch):
    #         self._save_model(epoch=self._current_epoch, batch=batch, logs=logs)
    #
    #
    #
    # def on_epoch_end(self, epoch, logs=None):
    #     if self.save_freq == "epoch":
    #         self._save_model(epoch=epoch, batch=None, logs=logs)
    #

    def should_save_on_step(self, cur_step: int) -> bool:
        """Handles batch-level saving logic, supports steps_per_execution."""
        if self.__save_freq == "epoch":
            return False
        else:
            if cur_step % self.__save_freq == 0:
                return True
            else:
                return False

    def save_model(self, save_obj: Dict[str, Any], step: int = None) -> None:
        cur_val: float = save_obj[self.__monitor]

        # Save best
        if self.__monitor_op(torch.Tensor([cur_val]), self.__best):
            # remove previous best epoch
            for name in os.listdir(self.__save_dir):
                if name.startswith("best"):
                    filepath = os.path.join(self.__save_dir, name)
                    os.remove(filepath)
                    break

            fname = f"best_epoch{self.cur_epoch}"
            fname += ".pt" if step is None else f"_step{step}.pt"
            fname = os.path.join(self.__save_dir, fname)
            torch.save(save_obj, fname)

            if self.__verbose:
                print(f"\nEpoch {self.cur_epoch}: {self.__monitor} improved from {self.__best:.5f} to {cur_val:.5f}")

            self.__best = cur_val
        else:
            if self.__verbose:
                print(f"Epoch {self.cur_epoch}: {self.__monitor} did not improve from {self.__best:.5f}")

        # Save by freq
        if not self.__save_best_only:
            fname = f"epoch{self.cur_epoch}"
            fname += ".pt" if step is None else f"_step{step}.pt"
            fname = os.path.join(self.__save_dir, fname)
            torch.save(save_obj, fname)

    def __repr__(self) -> str:
        return f"""Save dir: {self.__save_dir}
Monitor: {self.__monitor}
Save freq: {self.__save_freq}
Best: {self.__best}"""
