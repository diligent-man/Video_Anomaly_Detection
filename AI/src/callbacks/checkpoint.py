import copy
from typing import Dict, Any

from ..tools import Trainer
from ..utils import Checkpointer, make_border


__all__ = ["checkpoint_callbacks"]


def on_train_routine_start(instance: Trainer) -> None:
    checkpoint_config: Dict[str, Any] = copy.deepcopy(instance.config.Checkpoint.get_dict())
    checkpoint_config.pop("name")
    checkpoint_config.pop("apply")

    instance.checkpointer = Checkpointer(instance.config.Global.checkpoint_path, **checkpoint_config)

    top, bottom = make_border("Init checkpointer service")
    print(top)
    print(instance.checkpointer)
    print(bottom)


def on_train_epoch_start(instance: Trainer) -> None:
    instance.checkpointer.cur_epoch = instance.cur_epoch


def on_val_batch_end(instance: Trainer) -> None:
    if instance.checkpointer.should_save_on_step(instance.batch_output.step):
        save_obj: Dict[str, Any] = {
            "model": instance.model.state_dict() if instance.config.Checkpoint.save_weights_only else instance.model,
            "optimizer": instance.optimizer.state_dict(),
            "epoch": instance.batch_output.epoch,
            "step": instance.batch_output.step,
            "val_loss": instance.batch_output.as_metrics()["val_loss"]
        }

        if instance.config.Checkpoint.include_config:
            save_obj["config"] = instance.config

        instance.checkpointer.save_model(save_obj, instance.batch_output.step)


def on_val_epoch_end(instance: Trainer) -> None:
    save_obj: Dict[str, Any] = {
        "model": instance.model.state_dict() if instance.config.Checkpoint.save_weights_only else instance.model,
        "optimizer": instance.optimizer.state_dict(),
        "epoch": instance.batch_output.epoch,
        "step": instance.batch_output.step,
        "val_loss": instance.batch_output.as_metrics()["val_loss"]
    }

    if instance.config.Checkpoint.include_config:
        save_obj["config"] = instance.config

    instance.checkpointer.save_model(save_obj)


checkpoint_callbacks = {
    "on_train_routine_start": on_train_routine_start,
    "on_train_epoch_start": on_train_epoch_start,

    "on_val_batch_end": on_val_batch_end,
    "on_val_epoch_end": on_val_epoch_end,
    # "on_train_epoch_end": on_train_epoch_end,
    # "on_fit_epoch_end": on_fit_epoch_end,
    # "on_train_end": on_train_end,
}
