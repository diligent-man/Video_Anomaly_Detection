from tqdm import tqdm
from ...runner import Trainer
from .BaseCallback import BaseCallback


__all__ = ["ProgressCallback"]


class ProgressCallback(BaseCallback):
    """
    A [`TrainerCallback`] that displays the progress of training or evaluation.
    You can modify `max_str_len` to control how long strings are truncated when logging.
    """

    def __init__(self, max_str_len: int = 100):
        """
        Initialize the callback with optional max_str_len parameter to control string truncation length.

        Args:
            max_str_len (`int`):
                Maximum length of strings to display in logs.
                Longer strings will be truncated with a message.
        """
        self.training_bar = None
        self.prediction_bar = None
        self.max_str_len = max_str_len
        self.current_step = 0

    def on_train_begin(self, instance: Trainer) -> None:
        self.training_bar = tqdm(total=instance.state.steps, dynamic_ncols=True)

    def on_step_end(self, instance: Trainer) -> None:
        self.training_bar.update(instance.state.step)
        self.current_step = instance.state.step

    # def on_prediction_step(self, args, state, control, eval_dataloader=None, **kwargs):
    #     if state.is_world_process_zero and has_length(eval_dataloader):
    #         if self.prediction_bar is None:
    #             self.prediction_bar = tqdm(
    #                 total=len(eval_dataloader), leave=self.training_bar is None, dynamic_ncols=True
    #             )
    #         self.prediction_bar.update(1)
    #
    # def on_evaluate(self, args, state, control, **kwargs):
    #     if state.is_world_process_zero:
    #         if self.prediction_bar is not None:
    #             self.prediction_bar.close()
    #         self.prediction_bar = None
    #
    # def on_predict(self, args, state, control, **kwargs):
    #     if state.is_world_process_zero:
    #         if self.prediction_bar is not None:
    #             self.prediction_bar.close()
    #         self.prediction_bar = None
    #
    # def on_log(self, args, state, control, logs=None, **kwargs):
    #     if state.is_world_process_zero and self.training_bar is not None:
    #         # make a shallow copy of logs so we can mutate the fields copied
    #         # but avoid doing any value pickling.
    #         shallow_logs = {}
    #         for k, v in logs.items():
    #             if isinstance(v, str) and len(v) > self.max_str_len:
    #                 shallow_logs[k] = (
    #                     f"[String too long to display, length: {len(v)} > {self.max_str_len}. "
    #                     "Consider increasing `max_str_len` if needed.]"
    #                 )
    #             else:
    #                 shallow_logs[k] = v
    #         _ = shallow_logs.pop("total_flos", None)
    #         # round numbers so that it looks better in console
    #         if "epoch" in shallow_logs:
    #             shallow_logs["epoch"] = round(shallow_logs["epoch"], 2)
    #         self.training_bar.write(str(shallow_logs))

    def on_train_end(self, instance: Trainer):
        self.training_bar.close()
        self.training_bar = None
