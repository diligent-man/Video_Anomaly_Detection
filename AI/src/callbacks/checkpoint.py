import os
from typing import Dict, Any

import torch

from ..tools import Trainer
from ..utils import Checkpointer


__all__ = ["checkpoint_callbacks"]


def on_train_routine_start(instance: Trainer) -> None:
    instance.checkpointer = Checkpointer(instance.config.Global.checkpoint_path)

    if instance.config.Checkpoint.get("load", False):
        resume_name: str = instance.config.Global.resume_name
        assert os.path.isfile(resume_name), FileNotFoundError

        ckpt: Any = torch.load(f=resume_name, map_location="cpu")
        instance.start_epoch = ckpt["epoch"] + 1
        instance.model.load_state_dict(ckpt["model"].state_dict() if isinstance(ckpt["model"], torch.nn.Module) else ckpt["model"])
        instance.optimizer.load_state_dict(ckpt["optimizer"])
        del ckpt


def on_val_batch_end(instance: Trainer):
    if instance.config.Checkpoint.save and instance.batch_output.step % instance.config.Checkpoint.save_step == 0:
        obj: Dict[str, Any] = {
            "model": instance.model.state_dict() if instance.config.Checkpoint.save_weights_only else instance.model,
            "optimizer": instance.optimizer.state_dict(),
            "epoch": instance.batch_output.epoch,
            "step": instance.batch_output.step,
            "val_loss": instance.batch_output.as_metrics()["val_loss"]
        }

        if instance.config.Checkpoint.include_config:
            obj["config"] = instance.config

        print(instance.config.Checkpoint)

        # torch.save(obj, f"{os.path.join(instance.ckpt)}.pt")
        # torch.serialization.add_safe_globals([save_tensor])
        #
        #
        # self.__save_checkpoint(epoch=epoch, val_loss=run_epoch_result["loss"],
        #                        save_all=self.__config.CHECKPOINT_SAVE_ALL,
        #                        obj=obj
        #                        )


checkpoint_callbacks = {
    "on_val_batch_end": on_val_batch_end,
    # "on_train_epoch_end": on_train_epoch_end,
    # "on_fit_epoch_end": on_fit_epoch_end,
    # "on_train_end": on_train_end,
}
