from torch.optim.lr_scheduler import (
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

available_lr_scheduler = {
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
