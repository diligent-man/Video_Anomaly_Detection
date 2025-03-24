from typing import Tuple

from torch.utils.data import DataLoader

from ....runner import Trainer

__all__ = ["find_initial_total"]


def find_initial_total(instance: Trainer, dataloader: DataLoader) -> Tuple[int, int]:
    if instance.state.phase == "train" or instance.state.eval_strategy == "epoch":
        initial: int = instance.state.epoch - 1
        total: int = initial + instance.state.epochs

        initial *= len(dataloader)
        total *= len(dataloader)
    else:
        initial: int = instance.state.step // instance.state.eval_steps
        total: int = instance.state.steps // instance.state.eval_steps

        initial *= len(dataloader)
        total *= len(dataloader)
    return initial, total
