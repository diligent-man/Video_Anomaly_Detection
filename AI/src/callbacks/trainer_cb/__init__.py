from .BaseCallback import BaseCallback as TrainerCallback

from .Progress import Progress
from .DefaultFlow import DefaultFlow
from .Checkpointer import Checkpointer
from .EarlyStopping import EarlyStopping

# Should place with execution order awareness
DEFAULT_TRAINER_CALLBACKS = [
    DefaultFlow,
    Progress,
    Checkpointer,
    EarlyStopping
]


__all__ = [
    TrainerCallback,
    DEFAULT_TRAINER_CALLBACKS
]
