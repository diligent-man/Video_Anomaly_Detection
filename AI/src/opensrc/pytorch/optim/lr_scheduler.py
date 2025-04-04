from typing import Dict

from torch.optim.lr_scheduler import (
    LRScheduler,
    ChainedScheduler,
    ConstantLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ExponentialLR,
    LambdaLR,
    LinearLR,
    MultiplicativeLR,
    MultiStepLR,
    OneCycleLR,
    PolynomialLR,
    ReduceLROnPlateau,
    SequentialLR,
    StepLR
)


__all__ = ["avail_scheduler"]


avail_scheduler: Dict[str, LRScheduler] = {
    "LambdaLR": LambdaLR,
    "MultiplicativeLR": MultiplicativeLR,
    "StepLR": StepLR,
    "MultiStepLR": MultiStepLR,
    "ConstantLR": ConstantLR,
    "LinearLR": LinearLR,
    "ExponentialLR": ExponentialLR,
    "PolynomialLR": PolynomialLR,
    "CosineAnnealingLR": CosineAnnealingLR,
    "CosineAnnealingWarmRestarts": CosineAnnealingWarmRestarts,
    "ChainedScheduler": ChainedScheduler,
    "SequentialLR": SequentialLR,
    "ReduceLROnPlateau": ReduceLROnPlateau,
    "OneCycleLR": OneCycleLR
}
