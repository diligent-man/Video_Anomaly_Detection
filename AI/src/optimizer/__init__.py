from typing import Tuple, List, Iterator

from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler


from .optimizer import OPTIMIZERS
from .lr_scheduler import SCHEDULERS
from .regularizer import REGULARIZERS

from ..utils import DotDict, make_border
from ..modeling.architectures import VADDistillationModel


__all__ = ["build_optimizer", "OPTIMIZERS", "SCHEDULERS"]


def build_optimizer(config: DotDict,
                    model: Module
                    ) -> Tuple[Optimizer, None | LRScheduler]:
    top, bottom = make_border("Build optim")
    print(top)
    # step 1: build scheduler
    lr_config: DotDict = config.Optim.pop("lr", DotDict({}))
    name: None | str = lr_config.pop("name", None)
    assert name in OPTIMIZERS.keys(), ValueError(f"Invalid optimizer. Get '{name}'")

    if isinstance(model, VADDistillationModel):
        params: List[Tensor] = []
        for student in model.models["student"]:
            params += list(student.parameters())
    else:
        params: Iterator[Parameter] = model.parameters()
    optim: Optimizer = OPTIMIZERS[name](params, **lr_config.get_dict())

    # step 2: build regularization
    reg: None = None
    # regularizer_config = config.Optim.pop("regularizer", DotDict({}))
    # name = regularizer_config.pop("name", None)
    # if name is not None:
    #     assert name in REGULARIZERS, ValueError(f"Invalid regularizer. Get '{name}'")
    #     reg = REGULARIZERS[name](**regularizer_config.get_dict())

    # step 3: build scheduler
    scheduler: None = None
    scheduler_config: DotDict = config.Optim.pop("scheduler", DotDict({}))
    name: None | str = scheduler_config.pop("name", None)

    if name is not None:
        assert name in SCHEDULERS.keys(), ValueError(f"Invalid scheduler. Get '{name}'")
        scheduler: LRScheduler = SCHEDULERS[name](optim, **scheduler_config.get_dict())

    print(f"""Optimizer: {optim.__class__.__name__}
Scheduler: {scheduler if scheduler is None else scheduler.__class__.__name__}
Regularizer: {reg}""")
    print(bottom)
    return optim, scheduler
