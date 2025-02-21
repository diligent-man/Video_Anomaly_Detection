import inspect
from typing import Dict, Callable, Any, Union


import torch
from torch.utils.data import DataLoader


from . import Trainer
from ..losses import LossWrapper
from ..metrics import MetricWrapper
from .forward_strategy import FORWARD_STRATEGIES


__all__ = ["BatchForwarder"]


class BatchForwarder(object):
    """
    Forward model on a mini-batch manner. This class can be used in train/ val/ test phases
    """
    __epochs: int
    __cur_epoch: int

    def __init__(self,
                 epochs: int,
                 cur_epoch: int,
                 device: str
                 ) -> None:
        self.__epochs: int = epochs
        self.__cur_epoch: int = cur_epoch
        self.__device: str = device

    def __call__(self,
                 instance: Union[Trainer],
                 phase: str,
                 forward_strategy: str,
                 model: torch.nn.Module,
                 dataloader: DataLoader,
                 loss: LossWrapper,
                 metrics: MetricWrapper,
                 amp_cfg: Dict[str, Any],
                 optim: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                 grad_scaler: torch.GradScaler = None,
                 ) -> None:
        """
        Perform 1 epoch running with specific phase and selected forward strategy
        """
        assert phase in ("train", "val", "test"), ValueError("Selected phase is invalid")
        assert forward_strategy in FORWARD_STRATEGIES.keys(), ValueError(f"Selected strategy '{forward_strategy}' is not supported")

        forward_callable: Callable = FORWARD_STRATEGIES[forward_strategy]
        kwargs: Dict[str, Any] = {
            "instance": instance, "phase": phase, "epochs": self.__epochs, "cur_epoch": self.__cur_epoch,
            "grad_ctx_manager": torch.set_grad_enabled(phase == "train") if phase in ("train", "val") else torch.inference_mode(),
            "model": model, "dataloader": dataloader, "loss": loss, "metrics": metrics,
            "amp_cfg": amp_cfg, "grad_scaler": grad_scaler, "device": self.__device
        }
        if phase == "train":
            model.train()
            kwargs["optim"] = optim

            if "scheduler" in inspect.signature(forward_callable).parameters.keys():
                kwargs["scheduler"] = scheduler
        else:
            model.eval()

        forward_callable(**kwargs)
