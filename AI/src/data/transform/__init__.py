from typing import Any, Dict

from torchvision.transforms import Compose
from torchvision.transforms import InterpolationMode
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
    ToTensor
)

from .Permute import Permute

from ...opensrc.pytorch.Tensor import DTYPES


TRANSFORMS: Dict[str, Any] = {
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


__all__ = ["build_transforms"]


INTERPOLATIONS: Dict[str, Any] = {
    "NEAREST": InterpolationMode.NEAREST,
    "NEAREST_EXACT": InterpolationMode.NEAREST_EXACT,
    "BILINEAR": InterpolationMode.BILINEAR,
    "BICUBIC": InterpolationMode.BICUBIC,

    # For PIL compatibility
    "BOX": InterpolationMode.BOX,
    "HAMMING": InterpolationMode.HAMMING,
    "LANCZOS": InterpolationMode.LANCZOS,
}


def build_transforms(transforms: Dict[str, Dict] = None) -> Compose:
    compose: Compose = Compose([])
    if transforms is not None:
        # Verify transformation
        for transform in transforms.keys():
            assert transform in TRANSFORMS.keys(), "Your selected transform method is unavailable"

            # Verify interpolation mode & replace str name to its corresponding func
            if transform in ("Resize", "RandomRotation"):
                assert transforms[transform]["interpolation"] in INTERPOLATIONS.keys(), "Your selected interpolation mode in unavailable"
                transforms[transform]["interpolation"] = INTERPOLATIONS[transforms[transform]["interpolation"]]

            # Verify dtype & replace str name to its corresponding func
            if transform in ("ToDtype"):
                assert transforms[transform]["dtype"] in DTYPES.keys(), "Your selected dtype in unavailable"
                transforms[transform]["dtype"] = DTYPES[transforms[transform]["dtype"]]

            # if transform in ("Lambda"):
            #     assert transforms[transform]["lambd"] in available_callable.keys(), "Your selected callable in unavailable"
            #     transforms[transform]["lambd"] = available_callable[transforms[transform]["lambd"]]

        compose = Compose([TRANSFORMS[transform](**args) for transform, args in transforms.items()])
    return compose
