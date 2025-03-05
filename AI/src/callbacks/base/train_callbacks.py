from typing import Tuple

import torch

from AI.src.tools import Trainer
from AI.src.utils import ModelArchInspector


__all__ = [
    "on_train_routine_start",
    # "on_train_routine_end",
    # "on_train_start",
    # "on_train_epoch_start",
    # "on_train_batch_start",
    # "optimizer_step",
    # "on_before_zero_grad",
    # "on_train_batch_end",
    # "on_train_epoch_end",
    # "on_fit_epoch_end",
    # "on_model_save",
    # "on_train_end",
    # "on_params_update",
    # "teardown"
]


def on_train_routine_start(instance: Trainer):
    # Inspect model architecture
    inspect_model_arch: bool = instance.config.Global.get("inspect_model_arch", False)
    dummy_shape: None | Tuple[int, ...] = instance.config.Global.get("dummy_input_shape", None)

    if inspect_model_arch and dummy_shape is not None:
        try:
            with torch.amp.autocast(**instance.amp_config):
                model_arch = ModelArchInspector(
                    instance.model,
                    instance.config.Global.dummy_input_shape,
                    depth=instance.config.Global.get("inspect_depth", 3),
                    mode="train",
                    verbose=0
                )
            instance.config["Model_arch"] = model_arch
        except Exception as e:
            instance.config["Model_arch"] = f"Fail to inspect model architecture due to {e}"

# def on_train_routine_end(trainer): pass
# def on_train_start(trainer): pass
# def on_train_epoch_start(trainer): pass
# def on_train_batch_start(trainer): pass
# def optimizer_step(trainer): pass
# def on_before_zero_grad(trainer): pass
# def on_train_batch_end(trainer): pass
# def on_train_epoch_end(trainer): pass
# def on_fit_epoch_end(trainer): pass
# def on_model_save(trainer): pass
# def on_train_end(trainer): pass
# def on_params_update(trainer): pass
# def teardown(trainer): pass
