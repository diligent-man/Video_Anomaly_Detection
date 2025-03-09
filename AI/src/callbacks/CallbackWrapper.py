import warnings
from typing import List, Union, Any

import AI.src.runner.Trainer as Trainer  # due to cyclic dependency
from .trainer_cb import TrainerCallback
from ..utils.runner_utils.trainer import TrainerControl


__all__ = ["CallbackWrapper"]


class CallbackWrapper(object):
    """Internal class that just calls the list of callbacks in order."""
    def __init__(self,
                 instance: Union[Trainer],
                 integrated_callbacks: List[str],
                 **kwargs
                 ) -> None:
        self.__instance: Union[Trainer] = instance
        self.__callback_lst: List[Union[TrainerCallback]] = self._init_cb(integrated_callbacks)

        for k, v in kwargs.items():
            setattr(self, k, v)

    @property
    def callback_lst(self) -> List[Union[TrainerCallback]]:
        return self.__callback_lst

    def _init_cb(self, integrated_callbacks: List[str]) -> List[Union[TrainerCallback]]:
        callbacks_to_add: List[Union[type(TrainerCallback)]] = []

        # Trainer callbacks
        if isinstance(self.__instance, Trainer.Trainer):
            from .trainer_cb import DEFAULT_TRAINER_CALLBACKS
            callbacks_to_add: List = [*DEFAULT_TRAINER_CALLBACKS]

            for name in integrated_callbacks:
                if name == "mlflow":
                    from .intergrated_cb import MLflowCallback
                    callbacks_to_add.append(MLflowCallback)
        # Test callbacks

        # Inferer callbacks

        return_callbacks: List[Union[TrainerCallback]] = []
        for cb in callbacks_to_add:
            cb = cb() if isinstance(cb, type) else cb
            cb_class = cb if isinstance(cb, type) else cb.__class__

            if cb_class in [c.__class__ for c in return_callbacks]:
                warnings.warn(
                    f"You are adding a {cb_class} to the callbacks of this Trainer, but there is already one. The current" +
                    f"list of callbacks is\n:" +
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

    def __call__(self,
                 event: str,
                 control: Union[TrainerControl] = None,
                 **kwargs
                 ) -> Union[Any, TrainerControl]:
        for callback in self.__callback_lst:
            result: Any = getattr(callback, event)(
                self.__instance,
                **kwargs,
            )

            # A Callback can skip the return of 'control' if it doesn't change it.
            if result is not None:
                control = result
        return control
