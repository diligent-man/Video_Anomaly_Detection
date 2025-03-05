from typing import Union, List, Dict, Callable, Tuple

from .base import base_callbacks

from ..tools import Trainer
from ..utils import get_services


__all__ = [
    "add_callbacks",
    "SUPPORTED_CALLBACKS",
]


SUPPORTED_CALLBACKS: Tuple[str, ...] = (
    "checkpoint",
    "mlflow"
)


def add_callbacks(instance: Union[Trainer]) -> None:
    """
    Add integration callbacks from various sources to the instance's callbacks.

    Args:
        instance (Trainer, Predictor, Validator, Exporter): An object with a 'callbacks' attribute that is a dictionary
            of callback lists.
    """
    callbacks_lst: List[Dict[str, Callable]] = [base_callbacks]

    # Load training callbacks
    if instance.__class__.__name__ == "Trainer":
        for service in get_services(instance.config):
            assert service in SUPPORTED_CALLBACKS, ValueError(f"'{service}' is not supported in training phase")

            if service == "mlflow":
                from .mlflow import mlflow_callbacks
                callbacks_lst.append(mlflow_callbacks)
            elif service == "checkpoint":
                from .checkpoint import checkpoint_callbacks
                callbacks_lst.append(checkpoint_callbacks)

            # from .dvc import callbacks as dvc_cb
            # from .neptune import callbacks as neptune_cb
            # from .raytune import callbacks as tune_cb
            # from .tensorboard import callbacks as tb_cb
            # from .wb import callbacks as wb_cb

            # callbacks_list.extend([clear_cb, comet_cb, dvc_cb, mlflow_cb, neptune_cb, tune_cb, tb_cb, wb_cb])

    for callbacks in callbacks_lst:
        for k, v in callbacks.items():
            if v not in instance.callbacks[k]:
                instance.callbacks[k].append(v)
