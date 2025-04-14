from .BaseCallback import BaseCallback as TesterCallback

from .DefaultFlow import DefaultFlow


# Should place with execution order awareness
DEFAULT_TESTER_CALLBACKS = [
    DefaultFlow
]


__all__ = [
    TesterCallback,
    DEFAULT_TESTER_CALLBACKS
]
