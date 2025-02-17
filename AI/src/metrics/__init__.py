from typing import Dict, Callable, List, Any
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

from ..utils import DotDict, ANSIColor, make_border


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


__all__ = ["build_metric"]


def build_metric(config: DotDict) -> List[Metric]:
    top, bottom = make_border("Build metric")
    print(top)
    in_train: bool = config.Metric.pop("in_train", False)
    metrics: List[DotDict] = config.Metric.pop("metrics", [])
    assert len(metrics) > 0, ValueError("At least one metric is required")

    built_metrics: List[Metric] = []
    for i in range(len(metrics)):
        name: None | str = metrics[i].pop("name", None)
        args: Dict[str, Any] = metrics[i].get_dict()
        assert name in METRICS.keys(), ValueError(f"Metric {name} is invalid")

        metric: Metric = METRICS[name](**args)
        built_metrics.append(metric)
        print(f"Metric {i}: {metric.__class__.__name__}")
    print(f"Use metric during training: {in_train}")
    print(bottom)
    return built_metrics
