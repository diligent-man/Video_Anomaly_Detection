from .BaseCallback import BaseCallback as TrainerCallback

from .Progress import Progress
from .DefaultFlow import DefaultFlow
from .Checkpointer import Checkpointer

# Should place with execution order awareness
DEFAULT_TRAINER_CALLBACKS = [
    DefaultFlow,
    Progress,
    Checkpointer
]


__all__ = [
    TrainerCallback,
    DEFAULT_TRAINER_CALLBACKS
]
