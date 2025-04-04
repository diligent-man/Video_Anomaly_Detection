from typing import Dict

from torch.optim import (
    Optimizer,
    Adam,
    AdamW,
    NAdam,
    RAdam,
    SparseAdam,
    Adadelta,
    Adagrad,
    Adamax,
    ASGD,
    RMSprop,
    Rprop,
    LBFGS,
    SGD
)


__all__ = ["avail_optim"]


avail_optim: Dict[str, Optimizer] = {
    "Adam": Adam,
    "AdamW": AdamW,
    "NAdam": NAdam,
    "Adadelta": Adadelta,
    "Adagrad": Adagrad,
    "Adamax": Adamax,
    "RAdam": RAdam,
    "SparseAdam": SparseAdam,
    "RMSprop": RMSprop,
    "Rprop": Rprop,
    "ASGD": ASGD,
    "LBFGS": LBFGS,
    "SGD": SGD
}
