from tqdm import tqdm
from typing import Tuple

from torch.utils.data import DataLoader

from ...runner import Trainer
from .BaseCallback import BaseCallback
from ...utils.runner_utils.trainer import find_initial_total


__all__ = ["Progress"]


class Progress(BaseCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    You can modify `max_str_len` to control how long strings are truncated when logging.
    """
    __train_bar: tqdm = None
    __val_bar: tqdm = None

    __train_init_total: Tuple[int, int] = None
    __val_init_total: Tuple[int, int] = None

    def __init__(self, max_str_len: int = 100) -> None:
        """
        Initialize the callback with optional max_str_len parameter to control string truncation length.

        :param max_str_len:
            Maximum length of strings to display in logs. Longer strings will be truncated with a message.
        """
        super(Progress, self).__init__()
        self.__max_str_len: int = max_str_len

    # @staticmethod
    # def _find_initial_total(instance: Trainer, dataloader: DataLoader) -> Tuple[int, int]:
    #     if instance.state.phase == "train" or instance.state.eval_strategy == "epoch":
    #         initial: int = instance.state.epoch - 1
    #         total: int = initial + instance.state.epochs
    #
    #         initial *= len(dataloader)
    #         total *= len(dataloader)
    #     else:
    #         initial: int = instance.state.step // instance.state.eval_steps
    #         total: int = instance.state.steps // instance.state.eval_steps
    #
    #         initial *= len(dataloader)
    #         total *= len(dataloader)
    #     return initial, total

    @staticmethod
    def _make_desc(instance: Trainer, phase: str, loss: None | float) -> str:
        desc: str = f"Forward: {instance.config.Data[phase].forward_strategy}, " \
                    f"Phase: {phase}, " \
                    f"Loss: {loss}"
        return desc

    def _trigger_tqdm(self, instance: Trainer) -> None:
        if instance.state.batch_output is not None:
            phase: str = instance.state.batch_output.phase
            loss: float = instance.state.batch_output.loss
        else:
            phase = instance.state.phase
            loss: None = None

        dataloader = getattr(instance, f"{phase}_dataloader")
        attr_name = f"_{self.__class__.__name__}__{instance.state.phase}_init_total"

        if getattr(self, attr_name, None) is None:
            setattr(self, attr_name, find_initial_total(instance, dataloader))

            setattr(self,
                    f"_{self.__class__.__name__}__{phase}_bar",
                    tqdm(None,
                         self._make_desc(instance, phase, loss),
                         getattr(self, attr_name)[1],
                         initial=getattr(self, attr_name)[0],
                         dynamic_ncols=True,
                         colour="cyan" if phase == "train" else "yellow",
                         )
                    )

    def on_train_epoch_begin(self, instance: Trainer) -> None:
        self._trigger_tqdm(instance)

    def on_val_epoch_begin(self, instance: Trainer) -> None:
        self._trigger_tqdm(instance)

    def on_step_end(self, instance: Trainer) -> None:
        # self._trigger_tqdm(instance)
        attr_name: str = f"_{self.__class__.__name__}__{instance.state.batch_output.phase}_bar"

        if instance.state.batch_output is not None:
            phase: str = instance.state.batch_output.phase
            loss: float = instance.state.batch_output.loss
        else:
            phase = instance.state.phase
            loss: None = None

        getattr(self, attr_name).set_description_str(self._make_desc(instance, phase, loss), refresh=False)
        getattr(self, attr_name).update(1)
