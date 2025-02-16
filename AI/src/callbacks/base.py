# import copy
import collections
from typing import List, Union, Callable, Dict
from ..tools import Trainer
# from .val_callbacks import *
# from .train_callbacks import *


__all__ = [
    "get_default_callbacks",
    "add_integration_callbacks"
]


# DEFAULT_CALLBACKS = {
#     # Run in trainer
#     "on_pretrain_routine_start": [on_pretrain_routine_start],
#     "on_pretrain_routine_end": [on_pretrain_routine_end],
#     "on_train_start": [on_train_start],
#     "on_train_epoch_start": [on_train_epoch_start],
#     "on_train_batch_start": [on_train_batch_start],
#     "optimizer_step": [optimizer_step],
#     "on_before_zero_grad": [on_before_zero_grad],
#     "on_train_batch_end": [on_train_batch_end],
#     "on_train_epoch_end": [on_train_epoch_end],
#     "on_fit_epoch_end": [on_fit_epoch_end],  # fit = train + val
#     "on_model_save": [on_model_save],
#     "on_train_end": [on_train_end],
#     "on_params_update": [on_params_update],
#     "teardown": [teardown],
#
#     # Run in validator
#     "on_val_start": [on_val_start],
#     "on_val_batch_start": [on_val_batch_start],
#     "on_val_batch_end": [on_val_batch_end],
#     "on_val_end": [on_val_end],

    # Run in predictor
    # "on_predict_start": [on_predict_start],
    # "on_predict_batch_start": [on_predict_batch_start],
    # "on_predict_postprocess_end": [on_predict_postprocess_end],
    # "on_predict_batch_end": [on_predict_batch_end],
    # "on_predict_end": [on_predict_end],

    # Run in exporter
#     "on_export_start": [on_export_start],
#     "on_export_end": [on_export_end],
# }


# def get_default_callbacks() -> defaultdict:
#     """
#     Return a copy of the default_callbacks dictionary with lists as default values.
#
#     Returns:
#         (defaultdict): A defaultdict with keys from default_callbacks and empty lists as default values.
#     """
#     return defaultdict(list, copy.deepcopy(DEFAULT_CALLBACKS))


def add_callbacks(instance: Union[Trainer]) -> None:
    """
    Add integration callbacks from various sources to the instance's callbacks.

    Args:
        instance (Trainer, Predictor, Validator, Exporter): An object with a 'callbacks' attribute that is a dictionary
            of callback lists.
    """
    callbacks_lst: List[Dict[str, Callable]] = []

    # Load training callbacks
    if instance.__class__.__name__ == "Trainer":
        from .mlflow import callbacks as mlflow_callbacks
        # from .dvc import callbacks as dvc_cb
        # from .neptune import callbacks as neptune_cb
        # from .raytune import callbacks as tune_cb
        # from .tensorboard import callbacks as tb_cb
        # from .wb import callbacks as wb_cb

        # callbacks_list.extend([clear_cb, comet_cb, dvc_cb, mlflow_cb, neptune_cb, tune_cb, tb_cb, wb_cb])
        callbacks_lst += [mlflow_callbacks]

    for callbacks in callbacks_lst:
        for k, v in callbacks.items():
            if v not in instance.callbacks[k]:
                instance.callbacks[k].append(v)

#####################################################################################################
# Predictor callbacks --------------------------------------------------------------------------------------------------
# def on_predict_start(predictor): pass
# def on_predict_batch_start(predictor): pass
# def on_predict_batch_end(predictor): pass
# def on_predict_postprocess_end(predictor): pass
# def on_predict_end(predictor): pass
# Exporter callbacks ---------------------------------------------------------------------------------------------------
# def on_export_start(exporter): pass
# def on_export_end(exporter): pass