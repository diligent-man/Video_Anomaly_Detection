import dataclasses
from typing import List, Dict, Any


__all__ = ["BatchOutput"]


@dataclasses.dataclass
class BatchOutput:
    __phase: str
    __epoch: int
    __step: int

    __lr: float
    __loss: float
    __metric_names: List[str]
    __metric_values: float | List[float]

    def __init__(self,
                 phase: str,
                 epoch: int,
                 step: int,
                 lr: float = None,
                 loss: float = None,
                 metric_names: List[str] = None,
                 metric_values: float | List[float] = None
                 ) -> None:
        self.__phase = phase
        self.__epoch = epoch
        self.__step = step
        self.__lr = lr
        self.__loss = loss
        self.__metric_names = metric_names
        self.__metric_values = metric_values

    @property
    def step(self) -> int:
        return self.__step

    @property
    def epoch(self) -> int:
        return self.__epoch

    @property
    def phase(self):
        return self.__phase

    @property
    def loss(self) -> float:
        return self.__loss

    def to_dict(self) -> Dict[str, Any]:
        batch_output: Dict[str, Any] = {
            "phase": self.__phase,
            "epoch": self.__epoch,
            "step": self.__step,
            "lr": self.__lr,
            "loss": self.__loss,
            "metric_names": self.__metric_names,
            "metric_values": self.__metric_values
        }
        return batch_output

    def as_metrics(self) -> Dict[str, float]:
        """
        Only add single-scale metrics and loss
        """
        metrics: Dict[str, Any] = {
            f"{self.__phase}_loss": self.__loss
        }

        if self.__lr is not None:
            metrics["lr"] = self.__lr

        if self.__metric_names is not None:
            for i in range(len(self.__metric_names)):
                name: str = self.__metric_names[i]
                value: float | List[float] = self.__metric_values[i]

                if isinstance(value, float):
                    metrics[f"{self.__phase}_{name}"] = value
        return metrics
