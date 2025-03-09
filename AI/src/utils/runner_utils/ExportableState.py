__all__ = ["ExportableState"]


class ExportableState:
    """
    A class for objects that include the ability to have its state
    be saved during `Trainer._save_checkpoint` and loaded back in during
    `Trainer._load_from_checkpoint`.

    These must implement a `state` function that gets called during the respective
    Trainer function call. It should only include parameters and attributes needed to
    recreate the state at a particular time, to avoid utilizing pickle/maintain standard
    file IO writing.

    Example:

    ```python
    class EarlyStoppingCallback(TrainerCallback, ExportableState):
        def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0):
            self.early_stopping_patience = early_stopping_patience
            self.early_stopping_threshold = early_stopping_threshold
            # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
            self.early_stopping_patience_counter = 0

        def state(self) -> dict:
            return {
                "args": {
                    "early_stopping_patience": self.early_stopping_patience,
                    "early_stopping_threshold": self.early_stopping_threshold,
                },
                "attributes": {
                    "early_stopping_patience_counter": self.early_stopping_patience_counter,
                }
            }
    ```"""

    def state(self) -> dict:
        raise NotImplementedError("You must implement a `state` function to utilize this class.")
