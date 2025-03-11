from .BaseCallback import BaseCallback as TrainerCallback

from .Progress import ProgressCallback
from .DefaultFlowCallback import DefaultFlowCallback
from .CheckpointerCallback import CheckpointerCallback

# Should place with execution order awareness
DEFAULT_TRAINER_CALLBACKS = [
    DefaultFlowCallback,
    ProgressCallback,
    CheckpointerCallback
]


__all__ = [
    TrainerCallback,
    DEFAULT_TRAINER_CALLBACKS
]
