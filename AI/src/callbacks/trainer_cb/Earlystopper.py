from typing import Tuple

import torch

from ...runner import Trainer
from .BaseCallback import BaseCallback

__all__ = ["Earlystopper"]


class Earlystopper(BaseCallback):
    """
    This callback idea is adopted from Keras src
    """
    __support_monitor: Tuple[str] = "val_loss"  # currently
    __support_mode: Tuple[str] = "min"  # currently

    __counter = 0
    __best: float = None

    __patience: int
    __min_delta: float
    __check_from_epoch: int
    __verbose: bool

    def __init__(self,
                 mode: str = "min",
                 monitor: str = "val_loss",
                 patience: int = 5,
                 min_delta: float = 0.,
                 check_from_epoch: int = 0,
                 verbose: bool = True
                 ) -> None:
        assert mode in self.__support_mode, ValueError(f"Currently support {self.__support_mode}. Get {mode}")
        assert monitor in self.__support_monitor, ValueError(
            f"Currently support {self.__support_monitor}. Get {monitor}")
        super(Earlystopper, self).__init__()

        self.__mode = mode
        self.__monitor = monitor
        self.__patience = patience
        self.__min_delta = min_delta
        self.__check_from_epoch = check_from_epoch
        self.__verbose = verbose

        if mode == "min":
            self.__monitor_op = torch.less

            if self.__best is None:
                self.__best = torch.inf

    def _is_improved(self, new_val: float) -> torch.Tensor:
        return self.__monitor_op(torch.Tensor([new_val - self.__min_delta]), self.__best)

    def on_init_end(self, instance: Trainer) -> None:
        if instance.state.monitor is None:
            # New training
            instance.state.monitor = (self.__monitor, self.__best)
        else:
            # Resume best value from previous training if have
            self.__best = instance.state.monitor[1]

    def on_val_epoch_end(self, instance: Trainer) -> None:
        current_val: float = instance.state.batch_output.loss

        if instance.state.epoch >= self.__check_from_epoch:
            if self._is_improved(current_val):
                self.__best = current_val
                self.__counter = 0  # Restart wait if beat both the previous best

                # Overdrive value from State.monitor
                instance.state.monitor = (instance.state.monitor[0], self.__best)

                if self.__verbose:
                    print(f"{self.__monitor} is improved. Reset patience counter")
            else:
                self.__counter += 1
                if self.__verbose:
                    print(f"{self.__monitor} did not improved. Patience counter: {self.__counter}")

            if self.__counter >= self.__patience:
                # Patience has been exceeded: stop training
                instance.control.should_training_stop = True
                if self.__verbose:
                    print(f"Patience counter exceeds {self.__patience}")
