from typing import Dict, Any, List, Tuple, Union


import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling


from ..MLP import MLP
from ...utils import DotDict, create_feature_extractor

from .S3D import s3d
from .CLIP import clip_vision
from .InceptionI3D import inception_i3d


__all__ = ["build_backbone"]

backbones: Dict[str, Dict[str, Any]] = {
    "rgb_i3d": {
        "model": inception_i3d,
        "default_weight": "../weights/I3D/rgb.pt",
        "return_node": {"avg_pool": "features"},
        "dummy_input": [1, 3, 13, 224, 224]
    },

    "s3d": {
        "model": s3d,
        "default_weight": "../weights/S3D/model.pth",
        "return_node": {"avgpool": "features"},
        "dummy_input": [1, 3, 13, 224, 224]
    },

    "clip_vision": {
        "model": clip_vision,
        "default_weight": "../weights/CLIP/vit-base-patch16",
        "return_node": {"vision_model": "features"},
        "dummy_input": [1, 3, 224, 224],
    }
}


def build_backbone(config: DotDict) -> Union[Tuple[torch.nn.ModuleList, List[str], List[int]], \
                                             Tuple[torch.nn.ModuleList, List[str], torch.nn.ModuleList, List[int]]]:
    build_result: Dict[str, Any] = {
        "name": [],
        "backbone": torch.nn.ModuleList(),
        "out_channels": []
    }

    for name in config.Architecture.backbone.name:
        assert name in backbones.keys(), ValueError(f"Provided backbone is unavailable. Get '{name}'")
        build_result["name"].append(name)

        model_args = config.Architecture.backbone.get(f"{name}_args")
        model_args = {} if model_args is None else config.Architecture.backbone.get_dict(f"{name}_args")
        if model_args.get("weights") is None:
            model_args["weights"] = backbones[name]["default_weight"]

        model: torch.nn.Module = backbones[name]["model"](**model_args)
        model: torch.fx.GraphModule = create_feature_extractor(
            model, backbones[name]["return_node"],
            concrete_args=backbones[name].get("concrete_args")
        )
        model = _freeze_layer(model)
        model.eval()

        if config.Architecture.backbone.get("compile"):
            model.compile()

        out_channels: int = _get_out_channels(model, name)

        if config.Architecture.backbone.get("out_proj") is not None:
            out_proj = MLP(out_channels, **config.Architecture.backbone.get_dict("out_proj"))
            out_channels: int = out_proj.out_channels

            build_result["backbone"].append(model)
            build_result["out_channels"].append(out_channels)

            if "out_proj" not in build_result.keys():
                build_result["out_proj"] = torch.nn.ModuleList([out_proj])
            else:
                build_result["out_proj"].append(out_proj)
        else:
            build_result["backbone"].append(model)
            build_result["out_channels"].append(out_channels)
    return build_result["backbone"], build_result["name"], build_result.get("out_proj"), build_result["out_channels"]


def _get_out_channels(model: torch.nn.Module | torch.fx.GraphModule, name: str) -> int:
    dummy_input = torch.rand(backbones[name]["dummy_input"])
    output = model(dummy_input)["features"]

    # Rule-based approach due to various model output
    if issubclass(output.__class__, BaseModelOutputWithPooling):
        output = output[1]

    output_channel: int = output.squeeze().shape[0]
    return output_channel


def _freeze_layer(model: torch.nn.Module | torch.fx.GraphModule) -> torch.nn.Module | torch.fx.GraphModule:
    for para in reversed(list(model.parameters())):
        para.requires_grad = False
    return model
