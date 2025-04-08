from typing import Dict, Type
from torch.optim import (
    Optimizer,
    Adafactor,
    Adadelta,
    Adagrad,
    Adam,
    Adamax,
    AdamW,
    ASGD,
    LBFGS,
    NAdam,
    RAdam,
    RMSprop,
    Rprop,
    SGD,
    SparseAdam
)


OPTIMIZERS: Dict[str, Type[Optimizer]] = {
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
