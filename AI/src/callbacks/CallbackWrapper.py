import warnings
from functools import partial
from typing import List, Union, Any, Dict


import AI.src.runner.Trainer as Trainer  # due to cyclic dependency

from ..utils import DotDict
from .trainer_cb import TrainerCallback
from .trainer_cb import DEFAULT_TRAINER_CALLBACKS

from ..utils.runner_utils.trainer import TrainerControl


__all__ = ["CallbackWrapper"]


class CallbackWrapper(object):
    """Internal class that just calls the list of callbacks in order."""
    def __init__(self, instance: Union[Trainer], integrated_callbacks: List[str]) -> None:
        self.__instance: Union[Trainer] = instance
        self.__callback_lst: List[Union[TrainerCallback]] = self._init_cb(integrated_callbacks)

    @property
    def callback_lst(self) -> List[Union[TrainerCallback]]:
        return self.__callback_lst

    def _init_cb(self, integrated_callbacks: List[str]) -> List[Union[TrainerCallback]]:
        callbacks_to_add: List[Union[type(TrainerCallback)]] = []

        # Trainer callbacks
        if isinstance(self.__instance, Trainer.Trainer):
            callbacks_to_add: List = [*DEFAULT_TRAINER_CALLBACKS]

            for name in integrated_callbacks:
                if name == "mlflow":
                    from .intergrated_cb import MLflowCallback
                    callbacks_to_add.append(MLflowCallback)
        # Test callbacks

        # Inferer callbacks

        return_callbacks: List[Union[TrainerCallback]] = []

        for cb in callbacks_to_add:
            if cb.__name__ == "Checkpointer":
                checkpointer_config: Dict[str, Any] = self.__instance.config.get("Checkpointer", DotDict({})).get_dict()
                name, apply = checkpointer_config.pop("name"), checkpointer_config.pop("apply", True)
                if apply:
                    if "save_dir" not in checkpointer_config.keys():
                        checkpointer_config["save_dir"] = self.__instance.config.Global.ckpt_path

                    cb = cb(**checkpointer_config) if isinstance(cb, type) else partial(cb, **checkpointer_config)
                else:
                    continue
            else:
                cb = cb() if isinstance(cb, type) else cb

            cb_class = cb if isinstance(cb, type) else cb.__class__

            if cb_class in [c.__class__ for c in return_callbacks]:
                warnings.warn(
                    f"You are adding a {cb_class} to the callbacks of this Trainer, "
                    f"but there is already one. The current"
                    f"list of callbacks is\n:"
                    f"{return_callbacks}"
                )
            return_callbacks.append(cb)
        return return_callbacks

    # def pop_callback(self, callback):
    #     if isinstance(callback, type):
    #         for cb in self.callbacks:
    #             if isinstance(cb, callback):
    #                 self.callbacks.remove(cb)
    #                 return cb
    #     else:
    #         for cb in self.callbacks:
    #             if cb == callback:
    #                 self.callbacks.remove(cb)
    #                 return cb

    # def remove_callback(self, callback):
    #     if isinstance(callback, type):
    #         for cb in self.callbacks:
    #             if isinstance(cb, callback):
    #                 self.callbacks.remove(cb)
    #                 return
    #     else:
    #         self.callbacks.remove(callback)

    def insert_callback(self, callback: Union[TrainerCallback], idx: int) -> None:
        self.__callback_lst.insert(idx, callback)

    def __call__(self,
                 event: str,
                 control: Union[TrainerControl] = None,
                 **kwargs
                 ) -> Union[Any, TrainerControl]:
        for callback in self.__callback_lst:
            result: Any = getattr(callback, event)(self.__instance)

            # A Callback can skip the return of 'control' if it doesn't change it.
            if result is not None:
                control = result
        return control
