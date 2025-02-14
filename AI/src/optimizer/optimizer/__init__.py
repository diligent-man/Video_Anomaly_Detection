from typing import Dict, Callable
from torch.optim import (
    Adafactor,
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    ASGD,
    LBFGS,
    NAdam,
    Optimizer,
    RAdam,
    RMSprop,
    Rprop,
    SGD,
    SparseAdam
)


OPTIMIZERS: Dict[str, Callable] = {
    "Adafactor": Adafactor,
    "Adadelta": Adadelta,
    "Adagrad": Adagrad,
    "Adam": Adam,
    "Adamax": Adamax,
    "AdamW": AdamW,
    "ASGD": ASGD,
    "LBFGS": LBFGS,
    "NAdam": NAdam,
    "Optimizer": Optimizer,
    "RAdam": RAdam,
    "RMSprop": RMSprop,
    "Rprop": Rprop,
    "SGD": SGD,
    "SparseAdam": SparseAdam
}


__all__ = ["OPTIMIZERS"]
