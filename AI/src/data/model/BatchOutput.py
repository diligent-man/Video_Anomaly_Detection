import dataclasses
from typing import List, Dict, Any


__all__ = ["BatchOutput"]


@dataclasses.dataclass
class BatchOutput:
    __phase: str
    __lr: float
    __cur_step: int
    __loss: float
    __metric_names: List[str]
    __metric_values: float | List[float]

    def __init__(self,
                 phase: str,
                 cur_step: int,
                 lr: float = None,
                 loss: float = None,
                 metric_names: List[str] = None,
                 metric_values: float | List[float] = None
                 ) -> None:
        self.__phase = phase
        self.__lr = lr
        self.__cur_step = cur_step
        self.__loss = loss
        self.__metric_names = metric_names
        self.__metric_values = metric_values

    @property
    def step(self) -> int:
        return self.__cur_step

    def to_dict(self) -> Dict[str, Any]:
        batch_output: Dict[str, Any] = {
            "phase": self.__phase,
            "cur_step": self.__cur_step,
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

