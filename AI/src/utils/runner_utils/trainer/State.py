from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Tuple

from ....data.model import BatchOutput
from ..ExportableState import ExportableState


__all__ = ["State"]


@dataclass
class State:
    """
    A class containing the [`Trainer`] inner state that will be saved along the model and optimizer when checkpointing
    and passed to the [`TrainerCallback`].

    In all this class, one step is to be understood as one update step. When using gradient accumulation, one update
    step may require several forward and backward passes: if you use `gradient_accumulation_steps=n`, then one update
    step requires going through *n* batches.
    """
    phase: str = None

    step: int = 0
    epoch: int = 1

    steps: int = None   # derived from on_init_end of DefaultFlow callback
    epochs: int = None  # derived from on_init_end of DefaultFlow callback

    # logging_steps: int = None  # currently default at the end of train/ val
    eval_strategy: str = None  # derived from eval_strategy and in on_init_end of DefaultFlow callback
    eval_steps: str | int = None   # derived from eval_strategy and in on_init_end of DefaultFlow callback

    monitor: Tuple[str, float] = None  # derived when checkpointing or from  on_init_end of DefaultFlow callback

    batch_output: BatchOutput = None
    stateful_callbacks: List[ExportableState] | Dict[str, Any] = None

    def __post_init__(self) -> None:
        # Called after __init__()
        if self.stateful_callbacks is None:
            self.stateful_callbacks = {}
        elif isinstance(self.stateful_callbacks, dict):
            # We are loading the callbacks in from the state file, no need to process them
            pass
        else:
            # Saveable callbacks get stored as dict of kwargs
            stateful_callbacks = {}
            for callback in self.stateful_callbacks:
                if not isinstance(callback, ExportableState):
                    raise TypeError(
                        f"All callbacks passed to be saved must inherit `ExportableState`, but received {type(callback)}"
                    )

                name = callback.__class__.__name__

                if name in stateful_callbacks:
                    # We can have multiple versions of the same callback
                    # if so, we store them as a list of states to restore
                    if not isinstance(stateful_callbacks[name], list):
                        stateful_callbacks[name] = [stateful_callbacks[name]]
                    stateful_callbacks[name].append(callback.state())
                else:
                    stateful_callbacks[name] = callback.state()
            self.stateful_callbacks = stateful_callbacks
