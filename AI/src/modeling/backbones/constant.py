from typing import Dict, Any

import torch
from torch.nn import Module

from .S3D import s3d, S3D_Weights
from .CLIP import clip_vision, CLIP_Weights
from .InceptionI3D import inception_i3d, InceptionI3D_Weights

from ..nn import VectReducer


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
    "clip_vit/b16": {
        "model": clip_vision,
        "weights": CLIP_Weights.BASE_PATCH16_224,
        "return_node": {"vision_model": "features"},
        "dummy_input": [1, 3, 224, 224],
    },

    "clip_vit/b32": {
        "model": clip_vision,
        "weights": CLIP_Weights.BASE_PATCH32_224,
        "return_node": {"vision_model": "features"},
        "dummy_input": [1, 3, 224, 224],
    },

    "clip_vit/l14": {
        "model": clip_vision,
        "weights": CLIP_Weights.LARGE_PATCH14_224,
        "return_node": {"vision_model": "features"},
        "dummy_input": [1, 3, 224, 224],
    },

    "clip_vit/l14_336": {
        "model": clip_vision,
        "weights": CLIP_Weights.LARGE_PATCH14_336,
        "return_node": {"vision_model": "features"},
        "dummy_input": [1, 3, 336, 336],
    }
}

NET_2D = {"clip_vit/b16", "clip_vit/b32", "clip_vit/l14", "clip_vit/l14_336"}
NET_3D = {"rgb_i3d", "s3d"}

DEFAULT_2D_REDUCE = "mean"
DEFAULT_3D_REDUCE = "mean"


NET_2D_REDUCE: Dict[str, Module] = {
    "mean": torch.nn.AvgPool1d,

    "max_l1": VectReducer("max_l1"),
    "min_l1": VectReducer("min_l1"),

    "max_l2": VectReducer("max_l2"),
    "min_l2": VectReducer("min_l2"),

    "max_inf": VectReducer("max_inf"),
    "min_inf": VectReducer("min_inf"),

    "max_-inf": VectReducer("max_-inf"),
    "min_-inf": VectReducer("min_-inf"),
}

NET_3D_REDUCE: Dict[str, Module] = {
    "mean": torch.nn.AvgPool3d,

    "max_l1": VectReducer("max_l1"),
    "min_l1": VectReducer("min_l1"),

    "max_l2": VectReducer("max_l2"),
    "min_l2": VectReducer("min_l2"),

    "max_inf": VectReducer("max_inf"),
    "min_inf": VectReducer("min_inf"),

    "max_-inf": VectReducer("max_-inf"),
    "min_-inf": VectReducer("min_-inf"),
}
