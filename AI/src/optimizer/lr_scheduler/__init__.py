from typing import Dict, Callable
from torch.optim.lr_scheduler import (
    LambdaLR,
    MultiplicativeLR,
    StepLR,
    MultiStepLR,
    ConstantLR,
    LinearLR,
    ExponentialLR,
    SequentialLR,
    CosineAnnealingLR,
    ChainedScheduler,
    ReduceLROnPlateau,
    CyclicLR,
    CosineAnnealingWarmRestarts,
    OneCycleLR,
    PolynomialLR,
    LRScheduler
)

from .CustomExponentialLR import CustomExponentialLR
from .CustomSequentialLR import CustomSequentialLR
from .WarmupCosineAnnealingWarmRestarts import WarmupCosineAnnealingWarmRestarts


SCHEDULERS: Dict[str, Callable] = {
    "LambdaLR": LambdaLR,
    "MultiplicativeLR": MultiplicativeLR,
    "StepLR": StepLR,
    "MultiStepLR": MultiStepLR,
    "ConstantLR": ConstantLR,
    "LinearLR": LinearLR,
    "ExponentialLR": ExponentialLR,
    "SequentialLR": SequentialLR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "ChainedScheduler": ChainedScheduler,
    "ReduceLROnPlateau": ReduceLROnPlateau,
    "CyclicLR": CyclicLR,
    "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts,
    "OneCycleLR": OneCycleLR,
    "PolynomialLR": PolynomialLR,
    "LRScheduler": LRScheduler,
    "CustomExponentialLR": CustomExponentialLR,
    "CustomSequentialLR": CustomSequentialLR,
    "WarmupCosineAnnealingWarmRestarts": WarmupCosineAnnealingWarmRestarts,
}


__all__ = ["SCHEDULERS"]
