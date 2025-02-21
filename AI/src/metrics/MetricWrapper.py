from typing import List, Dict, Union, Any, Callable, Tuple
from collections.abc import Iterable


import torch

from multipledispatch import dispatch
from torcheval.metrics import (
    Metric,
    # Classification
    BinaryAccuracy,
    BinaryAUPRC,
    BinaryAUROC,
    BinaryBinnedAUPRC,
    BinaryBinnedAUROC,
    BinaryBinnedPrecisionRecallCurve,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryNormalizedEntropy,
    BinaryPrecision,
    BinaryPrecisionRecallCurve,
    BinaryRecall,
    BinaryRecallAtFixedPrecision,
    MulticlassAccuracy,
    MulticlassAUPRC,
    MulticlassAUROC,
    MulticlassBinnedAUPRC,
    MulticlassBinnedAUROC,
    MulticlassBinnedPrecisionRecallCurve,
    MulticlassConfusionMatrix,
    MulticlassF1Score,
    MulticlassPrecision,
    MulticlassPrecisionRecallCurve,
    MulticlassRecall,
    MultilabelAccuracy,
    MultilabelAUPRC,
    MultilabelBinnedAUPRC,
    MultilabelBinnedPrecisionRecallCurve,
    MultilabelPrecisionRecallCurve,
    MultilabelRecallAtFixedPrecision,
    TopKMultilabelAccuracy,
)


from ..utils import DotDict, make_border


METRICS: Dict[str, Callable] = {
    "BinaryAccuracy": BinaryAccuracy,
    "BinaryAUPRC": BinaryAUPRC,
    "BinaryAUROC": BinaryAUROC,
    "BinaryBinnedAUPRC": BinaryBinnedAUPRC,
    "BinaryBinnedAUROC": BinaryBinnedAUROC,
    "BinaryBinnedPrecisionRecallCurve": BinaryBinnedPrecisionRecallCurve,
    "BinaryConfusionMatrix": BinaryConfusionMatrix,
    "BinaryF1Score": BinaryF1Score,
    "BinaryNormalizedEntropy": BinaryNormalizedEntropy,
    "BinaryPrecision": BinaryPrecision,
    "BinaryPrecisionRecallCurve": BinaryPrecisionRecallCurve,
    "BinaryRecall": BinaryRecall,
    "BinaryRecallAtFixedPrecision": BinaryRecallAtFixedPrecision,
    "MulticlassAccuracy": MulticlassAccuracy,
    "MulticlassAUPRC": MulticlassAUPRC,
    "MulticlassAUROC": MulticlassAUROC,
    "MulticlassBinnedAUPRC": MulticlassBinnedAUPRC,
    "MulticlassBinnedAUROC": MulticlassBinnedAUROC,
    "MulticlassBinnedPrecisionRecallCurve": MulticlassBinnedPrecisionRecallCurve,
    "MulticlassConfusionMatrix": MulticlassConfusionMatrix,
    "MulticlassF1Score": MulticlassF1Score,
    "MulticlassPrecision": MulticlassPrecision,
    "MulticlassPrecisionRecallCurve": MulticlassPrecisionRecallCurve,
    "MulticlassRecall": MulticlassRecall,
    "MultilabelAccuracy": MultilabelAccuracy,
    "MultilabelAUPRC": MultilabelAUPRC,
    "MultilabelBinnedAUPRC": MultilabelBinnedAUPRC,
    "MultilabelBinnedPrecisionRecallCurve": MultilabelBinnedPrecisionRecallCurve,
    "MultilabelPrecisionRecallCurve": MultilabelPrecisionRecallCurve,
    "MultilabelRecallAtFixedPrecision": MultilabelRecallAtFixedPrecision,
    "TopKMultilabelAccuracy": TopKMultilabelAccuracy,
}


__all__ = ["MetricWrapper"]


class MetricWrapper(object):
    __metrics: List[Metric]
    __name: List[str]
    __in_train: bool

    def __init__(self, config: DotDict) -> None:
        self.__metrics, self.__names, self.__in_train = self._build_metric(config)

    @property
    def metrics(self) -> List[Metric]:
        return self.__metrics

    @property
    def name(self) -> List[str]:
        return self.__names

    @property
    def in_train(self) -> bool:
        return self.__in_train

    def update(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        # Check type except
        for metric in self.__metrics:
            try:
                metric.update(inputs, targets)
            except:
                if targets.dtype == torch.float:
                    targets = targets.type(torch.int)
                    metric.update(inputs, targets)
                elif targets.dtype == torch.int:
                    targets = targets.type(torch.float)
                    metric.update(inputs, targets)

    def compute(self) -> None:
        self.__metrics = [metric.compute() for metric in self.__metrics]

    def get_result(self) -> float | List[float]:
        # review type later
        return [_get_metric_result(metric) for metric in self.__metrics]

    @staticmethod
    def _build_metric(config: DotDict) -> Tuple[List[Metric], List[str], bool]:
        top, bottom = make_border("Build metric")
        print(top)
        in_train: bool = config.Metric.pop("in_train", False)
        metrics: List[DotDict] = config.Metric.pop("metrics", [])
        assert len(metrics) > 0, ValueError("At least one metric is required")

        names: List[str] = []
        built_metrics: List[Metric] = []
        for i in range(len(metrics)):
            name: None | str = metrics[i].pop("name", None)
            args: Dict[str, Any] = metrics[i].get_dict()
            assert name in METRICS.keys(), ValueError(f"Metric {name} is invalid")

            metric: Metric = METRICS[name](**args)

            names.append(name)
            built_metrics.append(metric)
            print(f"Metric {i}: {metric.__class__.__name__}")
        print(f"Use metric during training: {in_train}")
        print(bottom)
        return built_metrics, names, in_train


@dispatch(torch.Tensor)
def _get_metric_result(computed_metric: torch.Tensor) -> Union[float, List[float]]:
    return computed_metric.item() if computed_metric.dim() == 1 and len(computed_metric) == 1 else computed_metric.detach().cpu().numpy().tolist()


@dispatch(Iterable)
def _get_metric_result(computed_metric: Iterable) -> List[float]:
    result = []
    for constituent in computed_metric:
        if isinstance(constituent, List):
            result.append(_get_metric_result(constituent))
        else:
            if constituent.dim() == 0:
                result.append(_get_metric_result(constituent))
            else:
                constituent_result = []
                for tensor in constituent:
                    constituent_result.append(_get_metric_result(tensor))
                result.append(constituent_result)
    return result
