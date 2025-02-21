import dataclasses
from typing import List, Dict, Any


__all__ = ["BatchOutput"]


@dataclasses.dataclass
class BatchOutput:
    phase: str
    iteration: int
    loss: float
    metric_names: List[str]
    metric_values: float | List[float]

    def __init__(self,
                 phase: str,
                 iteration: int,
                 loss: float,
                 metric_names: List[str],
                 metric_values: float | List[float]
                 ) -> None:
        self.phase = phase
        self.iteration = iteration
        self.loss = loss
        self.metric_names = metric_names
        self.metric_values = metric_values

    def to_dict(self) -> Dict[str, Any]:
        batch_output: Dict[str, Any] = {
            "phase": self.phase,
            "iter": self.iteration,
            "loss": self.loss,
            "metric_names": self.metric_names,
            "metric_values": self.metric_values
        }
        return batch_output
