from .BaseCallback import BaseCallback as TrainerCallback

from .Progress import Progress
from .DefaultFlow import DefaultFlow
from .Checkpointer import Checkpointer
from .Earlystopper import Earlystopper

# Should place with execution order awareness
DEFAULT_TRAINER_CALLBACKS = [
    DefaultFlow,
    Progress,
    Checkpointer,
    Earlystopper
]


__all__ = [
    TrainerCallback,
    DEFAULT_TRAINER_CALLBACKS
]
