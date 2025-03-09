import inspect
from inspect import Parameter
from typing import Dict, Callable, Any, Union, List

import torch
from torch.autograd.grad_mode import set_grad_enabled, inference_mode

import AI.src.runner.Trainer as Trainer  # Circular import
from ..utils import get_amp_cfg
from ..utils.forward_strategy import FORWARD_STRATEGIES


__all__ = ["BatchForwarder"]


class BatchForwarder(object):
    """
    Forward model on a mini-batch manner. This class can be used in train/ val/ test phases
    """
    def __init__(self, forward_strategy: str, instance: Union[Trainer], **kwargs) -> None:
        if isinstance(instance, Trainer.Trainer):
            assert instance.state.phase in ("train", "val"), ValueError("Selected phase is invalid")

        assert forward_strategy in FORWARD_STRATEGIES.keys(), ValueError(
            f"Selected strategy '{forward_strategy}' is not supported"
        )

        super(BatchForwarder, self).__init__()

        self.__forward_fn: Callable = FORWARD_STRATEGIES[forward_strategy]
        instance.model.train() if instance.state.phase == "train" else instance.model.eval()

        self.__args: Dict[str, Any] = self._prep_args(self.__forward_fn, instance, **kwargs)

    @staticmethod
    def _prep_args(forward_fn: Callable, instance: Union[Trainer], **kwargs) -> Dict[str, Any]:
        amp_cfg, grad_scaler = get_amp_cfg(instance.config)
        grad_ctx: set_grad_enabled | inference_mode = torch.set_grad_enabled(instance.state.phase == "train")\
            if instance.state.phase in ("train", "val") else torch.inference_mode()

        args: Dict[str, Any] = {
            # Compulsory paras
            "instance": instance,
            "grad_ctx": grad_ctx,
            "dataloader": getattr(instance, f"{instance.state.phase}_dataloader"),
            "amp_cfg": amp_cfg,
            "grad_scaler": grad_scaler,

            # Optional paras
            "loss": getattr(instance, "loss", None),
            "metric": instance.metric if instance.config.Metric[f"in_{instance.state.phase}"] else None,

            # Only available in train phase
            "optim": instance.optim if instance.state.phase == "train" else None,
            "scheduler": instance.scheduler if instance.state.phase == "train" else None,
        }

        overridden_args: Dict[str, Any] = kwargs.pop("overridden_args", {})
        assert len(set(overridden_args.keys()).intersection(set(args.keys()))) == 0, ValueError(
            f"Overridden args must not include {args.keys()} keys"
        )

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

    def __call__(self) -> None:
        """Perform 1 epoch running with specific phase and selected forward strategy"""
        self.__forward_fn(**self.__args)
