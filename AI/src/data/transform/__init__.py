from typing import Dict, List, Callable
from torchvision.transforms.v2 import (
    # Color
    ColorJitter,
    Grayscale,
    RandomAdjustSharpness,
    RandomAutocontrast,
    RandomChannelPermutation,
    RandomEqualize,
    RandomGrayscale,
    RandomInvert,
    RandomPhotometricDistort,
    RandomPosterize,
    RandomSolarize,

    # Geometry
    CenterCrop,
    ElasticTransform,
    FiveCrop,
    Pad,
    RandomAffine,
    RandomCrop,
    RandomHorizontalFlip,
    RandomIoUCrop,
    RandomPerspective,
    RandomResize,
    RandomResizedCrop,
    RandomRotation,
    RandomShortestSize,
    RandomVerticalFlip,
    RandomZoomOut,
    Resize,
    ScaleJitter,
    TenCrop,

    # Meta
    ClampBoundingBoxes,
    ConvertBoundingBoxFormat,

    # Misc
    ConvertImageDtype,
    GaussianBlur,
    Identity,
    Lambda,
    LinearTransformation,
    Normalize,
    SanitizeBoundingBoxes,
    ToDtype,

    # Temporal
    UniformTemporalSubsample,

    # Type conversion
    PILToTensor,
    ToImage,
    ToPILImage,
    ToPureTensor,
    ToTensor,

    # Misc
    Compose,
    Transform,
    InterpolationMode
)

from .Permute import Permute
from ...opensrc.pytorch.Tensor import DTYPES


TRANSFORMS: Dict[str, Callable] = {
    # Color
    "ColorJitter": ColorJitter,
    "Grayscale": Grayscale,
    "RandomAdjustSharpness": RandomAdjustSharpness,
    "RandomAutocontrast": RandomAutocontrast,
    "RandomChannelPermutation": RandomChannelPermutation,
    "RandomEqualize": RandomEqualize,
    "RandomGrayscale": RandomGrayscale,
    "RandomInvert": RandomInvert,
    "RandomPhotometricDistort": RandomPhotometricDistort,
    "RandomPosterize": RandomPosterize,
    "RandomSolarize": RandomSolarize,

    # Geometry
    "CenterCrop": CenterCrop,
    "ElasticTransform": ElasticTransform,
    "FiveCrop": FiveCrop,
    "Pad": Pad,
    "RandomAffine": RandomAffine,
    "RandomCrop": RandomCrop,
    "RandomHorizontalFlip": RandomHorizontalFlip,
    "RandomIoUCrop": RandomIoUCrop,
    "RandomPerspective": RandomPerspective,
    "RandomResize": RandomResize,
    "RandomResizedCrop": RandomResizedCrop,
    "RandomRotation": RandomRotation,
    "RandomShortestSize": RandomShortestSize,
    "RandomVerticalFlip": RandomVerticalFlip,
    "RandomZoomOut": RandomZoomOut,
    "Resize": Resize,
    "ScaleJitter": ScaleJitter,
    "TenCrop": TenCrop,

    # Meta
    "ClampBoundingBoxes": ClampBoundingBoxes,
    "ConvertBoundingBoxFormat": ConvertBoundingBoxFormat,

    # Misc
    "ConvertImageDtype": ConvertImageDtype,
    "GaussianBlur": GaussianBlur,
    "Identity": Identity,
    "Lambda": Lambda,
    "LinearTransformation": LinearTransformation,
    "Normalize": Normalize,
    "SanitizeBoundingBoxes": SanitizeBoundingBoxes,
    "ToDtype": ToDtype,

    # Temporal
    "UniformTemporalSubsample": UniformTemporalSubsample,

    # Type conversion
    "PILToTensor": PILToTensor,
    "ToImage": ToImage,
    "ToPILImage": ToPILImage,
    "ToPureTensor": ToPureTensor,
    "ToTensor": ToTensor,

    # Custom
    "Permute": Permute
}


__all__ = ["build_transform"]


INTERPOLATIONS: Dict[str, InterpolationMode] = {
    "NEAREST": InterpolationMode.NEAREST,
    "NEAREST_EXACT": InterpolationMode.NEAREST_EXACT,
    "BILINEAR": InterpolationMode.BILINEAR,
    "BICUBIC": InterpolationMode.BICUBIC,

    # For PIL compatibility
    "BOX": InterpolationMode.BOX,
    "HAMMING": InterpolationMode.HAMMING,
    "LANCZOS": InterpolationMode.LANCZOS,
}


def _preprocess_duplicate(transform: str) -> str:
    """
    :param transform: name of transform with/ without suffix for repetition
    :return: preprocessed transform
    """
    if "_" in transform:
        transform = transform.split("_")[0]
    return transform


def build_transform(transform_cfg: None | Dict[str, Dict] = None) -> None | Compose:
    if len(transform_cfg.keys()) == 0 or transform_cfg is None:
        return None

    # Verify & init transformation
    compose: List[Transform] = []
    for transform, args in transform_cfg.items():
        transform: str = _preprocess_duplicate(transform)
        assert transform in TRANSFORMS.keys(), "Your selected transform method is unavailable"

        # Verify interpolation mode & replace str name to its corresponding func
        if transform in ("Resize", "RandomRotation"):
            assert transform_cfg[transform]["interpolation"] in INTERPOLATIONS.keys(), "Your selected interpolation mode in unavailable"
            transform_cfg[transform]["interpolation"] = INTERPOLATIONS[transform_cfg[transform]["interpolation"]]

        # Verify dtype & replace str name to its corresponding func
        if transform in "ToDtype":
            assert transform_cfg[transform]["dtype"] in DTYPES.keys(), "Your selected dtype in unavailable"
            transform_cfg[transform]["dtype"] = DTYPES[transform_cfg[transform]["dtype"]]

        # TODO: Add transform for lambda fn
        # if transform in ("Lambda"):
        #     assert transforms[transform]["lambd"] in available_callable.keys(), "Your selected callable in unavailable"
        #     transforms[transform]["lambd"] = available_callable[transforms[transform]["lambd"]]
        compose.append(TRANSFORMS[transform](**args))
    compose: Compose = Compose(compose)
    return compose
