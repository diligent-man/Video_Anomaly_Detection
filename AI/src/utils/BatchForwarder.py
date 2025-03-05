import inspect
from inspect import Parameter
from typing import Dict, Callable, Any, Union, List

import torch
from torch.nn import Module
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from AI.src.tools import Trainer
from AI.src.losses import LossWrapper
from AI.src.metrics import MetricWrapper
from AI.src.utils.forward_strategy import FORWARD_STRATEGIES


__all__ = ["BatchForwarder"]


class BatchForwarder(object):
    """
    Forward model on a mini-batch manner. This class can be used in train/ val/ test phases
    """

    def __init__(self, device: str = "cpu") -> None:
        super(BatchForwarder, self).__init__()
        self.__device: str = device

    def _prep_args(self,
                   forward_fn: Callable,
                   instance: Union[Trainer],
                   phase: str,
                   model: Module,
                   dataloader: DataLoader,
                   amp_cfg: Dict[str, Any],
                   loss: LossWrapper = None,
                   metrics: MetricWrapper = None,
                   optim: Optimizer = None,
                   scheduler: LRScheduler = None,
                   grad_scaler: torch.GradScaler = None,
                   **kwargs
                   ) -> Dict[str, Any]:
        args: Dict[str, Any] = {
            "instance": instance, "phase": phase, "model": model, "dataloader": dataloader, "amp_cfg": amp_cfg,
            "loss": loss, "metrics": metrics, "optim": optim, "scheduler": scheduler, "grad_scaler": grad_scaler,

            "epochs": kwargs.pop("epochs", 1),
            "cur_epoch": kwargs.pop("cur_epoch", 1),
            "device": self.__device,
            "grad_ctx_manager": torch.set_grad_enabled(phase == "train") if phase in ("train", "val") else torch.inference_mode(),
        }

        overridden_args: Dict[str, Any] = kwargs.pop("overridden_args", {})
        assert len(set(overridden_args.keys()).intersection(set(args.keys()))) == 0, ValueError(f"Overridden args must not include {args.keys()} keys")

        args = {**args, **overridden_args}

        # Update fn paras
        fn_paras: List[Parameter] = list(inspect.signature(forward_fn).parameters.values())
        for i in range(len(fn_paras)):
            para: Parameter = fn_paras[i]
            if para.default is not None:
                assert para.name in args.keys(), ValueError(f"{para.name} is not provided")

            if para.name in args.keys():
                fn_paras[i] = para.replace(default=args[para.name])
        return {para.name: para.default for para in fn_paras}

    def __call__(self,
                 forward_strategy: str,
                 instance: Union[Trainer],
                 phase: str,
                 model: Module,
                 dataloader: DataLoader,
                 amp_cfg: Dict[str, Any],
                 loss: LossWrapper = None,
                 metrics: MetricWrapper = None,
                 optim: Optimizer = None,
                 scheduler: LRScheduler = None,
                 grad_scaler: torch.GradScaler = None,
                 **kwargs
                 ) -> None:
        """
        Perform 1 epoch running with specific phase and selected forward strategy
        """
        assert phase in ("train", "val", "test"), ValueError("Selected phase is invalid")
        assert forward_strategy in FORWARD_STRATEGIES.keys(), ValueError(f"Selected strategy '{forward_strategy}' is not supported")

        forward_fn: Callable = FORWARD_STRATEGIES[forward_strategy]
        model.train() if phase == "train" else model.eval()

        args: Dict[str, Any] = self._prep_args(
            forward_fn,
            instance,
            phase,
            model,
            dataloader,
            amp_cfg,
            loss,
            metrics,
            optim,
            scheduler,
            grad_scaler,
            **kwargs
        )

        forward_fn(**args)
