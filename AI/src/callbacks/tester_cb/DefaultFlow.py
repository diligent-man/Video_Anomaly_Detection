import os
import pathlib
from typing import Dict, Any


import torch


from ...runner import Tester
from .BaseCallback import BaseCallback


__all__ = ["DefaultFlow"]


class DefaultFlow(BaseCallback):
    """
    A [`TrainerCallback`] that handles the default flow of the training loop for logs, evaluation and checkpoints.
    """
    def on_init_end(self, instance: Tester) -> None:
        # point directly to checkpoint
        resume_ckpt: str = str(pathlib.Path(instance.config.Global.get("resume_ckpt", "")))
        assert os.path.isfile(resume_ckpt), FileNotFoundError

        ckpt: Dict[str, Any] | torch.nn.Module = torch.load(f=resume_ckpt, map_location="cpu")

        # Load model's weights
        if isinstance(ckpt["model"], dict):
            instance.model.load_state_dict(ckpt["model"])
        else:
            instance.model.load_state_dict(ckpt["model"].state_dict())
        del ckpt

    def on_begin(self, instance: Tester) -> None:
        instance.state.phase = "test"

        if os.path.exists(os.path.join(instance.config.Global.log_path, "pred_result.txt")):
            os.remove(os.path.join(instance.config.Global.log_path, "pred_result.txt"))

    def on_step_begin(self, instance: Tester) -> None:
        pass

    def on_step_end(self, instance: Tester) -> None:
        instance.logger.write(
            os.path.join(instance.config.Global.log_path, "pred_result.txt"),
            f"{instance.state.step_info}\n",
            "a"
        )

    def on_end(self, instance: Tester) -> None:
        instance.logger.write(
            os.path.join(instance.config.Global.log_path, "metric_result.txt"),
            instance.state.metric_result,
            "a"
        )
