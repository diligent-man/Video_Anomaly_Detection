from typing import Dict, Callable
from .train_callbacks import *


base_callbacks: Dict[str, Callable] = {
    "on_train_routine_start": on_train_routine_start,
    # "on_pretrain_routine_end": [on_pretrain_routine_end],
    # "on_train_start": [on_train_start],
    # "on_train_epoch_start": [on_train_epoch_start],
    # "on_train_batch_start": [on_train_batch_start],
    # "optimizer_step": [optimizer_step],
    # "on_before_zero_grad": [on_before_zero_grad],
    # "on_train_batch_end": [on_train_batch_end],
    # "on_train_epoch_end": [on_train_epoch_end],
    # "on_fit_epoch_end": [on_fit_epoch_end],  # fit = train + val
    # "on_model_save": [on_model_save],
    # "on_train_end": [on_train_end],
    # "on_params_update": [on_params_update],
    # "teardown": [teardown],

    # "on_val_start": [on_val_start],
    # "on_val_batch_start": [on_val_batch_start],
    # "on_val_batch_end": [on_val_batch_end],
    # "on_val_end": [on_val_end],

    # "on_predict_start": [on_predict_start],
    # "on_predict_batch_start": [on_predict_batch_start],
    # "on_predict_postprocess_end": [on_predict_postprocess_end],
    # "on_predict_batch_end": [on_predict_batch_end],
    # "on_predict_end": [on_predict_end],

    # "on_export_start": [on_export_start],
    # "on_export_end": [on_export_end],
}


__all__ = [
    base_callbacks
]
