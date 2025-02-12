from typing import Dict, Any

import torch

from .CLIP import clip_vision
from .S3D import s3d, S3D_Weights
from .InceptionI3D import inception_i3d, InceptionI3D_Weights

from ..nn import MiddleSelector


__all__ = [
    "NET_DEFAULT_CONFIG",
    "NET_2D", "NET_3D",
    "NET_2D_REDUCE", "NET_3D_REDUCE",
    "DEFAULT_2D_REDUCE", "DEFAULT_3D_REDUCE"
]

NET_DEFAULT_CONFIG: Dict[str, Dict[str, Any]] = \
    {
    # These models are from pytorch code base
    "rgb_i3d": {
        "model": inception_i3d,
        "weights": InceptionI3D_Weights.DEFAULT,
        "return_node": {"avg_pool": "features"},
        "dummy_input": [1, 3, 13, 224, 224]
    },

    "s3d": {
        "model": s3d,
        "weights": S3D_Weights.DEFAULT,
        "return_node": {"avgpool": "features"},
        "dummy_input": [1, 3, 13, 224, 224]
    },

    # These models are from huggingface code base
    "clip_vision": {
        "model": clip_vision,
        "weights": "../weights/CLIP/vit-base-patch16",
        "return_node": {"vision_model": "features"},
        "dummy_input": [1, 3, 224, 224],
        # "concrete_args": {"return_dict": None},
    }
}

NET_2D = {"clip_vision"}
NET_3D = {"rgb_i3d", "s3d"}

DEFAULT_2D_REDUCE = "max"
DEFAULT_3D_REDUCE = "max"

NET_2D_REDUCE: Dict[str, Any] = {
    "max": torch.nn.MaxPool1d,
    "mid": MiddleSelector,
}

NET_3D_REDUCE: Dict[str, Any] = {
    "max": torch.nn.MaxPool3d,
    "avg": torch.nn.AvgPool3d,
    "fractional_max": torch.nn.FractionalMaxPool3d,
    "lp": torch.nn.LPPool3d,
}
