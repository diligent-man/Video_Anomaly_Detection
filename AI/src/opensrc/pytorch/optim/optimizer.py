from torch.optim import (
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

available_optimizers = {
    "Adam": Adam,
    "AdamW": AdamW,
    "NAdam": NAdam,
    "Adadelta": Adadelta,
    "Adagrad": Adagrad,
    "Adamax": Adamax,
    "RAdam": RAdam,
    "SparseAdam":SparseAdam,
    "RMSprop": RMSprop,
    "Rprop": Rprop,
    "ASGD": ASGD,
    "LBFGS": LBFGS,
    "SGD": SGD
}
