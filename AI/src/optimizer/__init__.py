from typing import Tuple

import torch

from .optimizer import OPTIMIZERS
from .lr_scheduler import SCHEDULERS
from .regularizer import REGULARIZERS

from ..utils import DotDict, ANSIColor


__all__ = ["build_optimizer", "OPTIMIZERS", "SCHEDULERS"]


def build_optimizer(config: DotDict,
                    model: torch.nn.Module
                    ) -> Tuple[torch.optim.Optimizer, None | torch.optim.lr_scheduler.LRScheduler]:
    print(f"{ANSIColor().CYAN}--------------- Building optimizer ---------------{ANSIColor().RESET}")
    # step1 build scheduler
    lr_config: DotDict = config.Optim.pop("lr", DotDict({}))
    name: None | str = lr_config.pop("name", None)
    assert name in OPTIMIZERS.keys(), ValueError(f"Invalid optimizer. Get '{name}'")

    optim: torch.optim.Optimizer = OPTIMIZERS[name](model.parameters(), **lr_config.get_dict())

    # step2 build regularization
    reg: None = None
    # regularizer_config = config.Optim.pop("regularizer", DotDict({}))
    # name = regularizer_config.pop("name", None)
    # if name is not None:
    #     assert name in REGULARIZERS, ValueError(f"Invalid regularizer. Get '{name}'")
    #     reg = REGULARIZERS[name](**regularizer_config.get_dict())

    # step3 build scheduler
    scheduler: None = None
    scheduler_config: DotDict = config.Optim.pop("scheduler", DotDict({}))
    name: None | str = lr_config.pop("name", None)

    if name is not None:
        assert name in SCHEDULERS.keys(), ValueError(f"Invalid scheduler. Get '{name}'")
        scheduler: torch.optim.lr_scheduler.LRScheduler = SCHEDULERS[name](optim, **scheduler_config.get_dict())

    print(f"""Optimizer: {optim.__class__.__name__}
Scheduler: {scheduler if scheduler is None else scheduler.__class__.__name__}
Regularizer: {reg}
{ANSIColor().CYAN}--------------------------------------------------{ANSIColor().RESET}""")
    return optim, scheduler
