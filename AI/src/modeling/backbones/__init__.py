import functools
from typing import Dict, Any, List, Tuple, Union

import torch
from transformers.modeling_outputs import BaseModelOutputWithPooling

from .ModelForwarder import ModelForwarder
from .constant import (
    NET_DEFAULT_CONFIG,
    NET_2D, NET_3D,
    NET_2D_REDUCE, NET_3D_REDUCE,
    DEFAULT_2D_REDUCE, DEFAULT_3D_REDUCE
)

from ..nn import MLP
from ...utils import DotDict, create_feature_extractor


__all__ = [
    "build_backbone",
    "ModelForwarder",
    "NET_DEFAULT_CONFIG",
    "NET_2D", "NET_3D",
    "NET_2D_REDUCE", "NET_3D_REDUCE",
    "DEFAULT_2D_REDUCE", "DEFAULT_3D_REDUCE"
]


def build_backbone(config: DotDict) -> Union[Tuple[List[torch.nn.Module], List[str], List[torch.nn.Module], List[int]],
                                             Tuple[torch.nn.ModuleList, List[str], List[torch.nn.Module], torch.nn.ModuleList, List[int]]]:
    build_result: Dict[str, Any] = {
        "name": [],
        "backbone": torch.nn.ModuleList(),
        "reduce": [],
        "out_channels": []
    }
    names: None | List[str] = config.Architecture.backbone.pop("name", None)
    compile_model: bool = config.Architecture.backbone.pop("compile", False)
    out_proj: Dict[str, Any] = config.Architecture.backbone.pop("out_proj", DotDict({})).get_dict()

    assert isinstance(names, list), ValueError(f"Name for feat extractor backbone must be string list. Get '{type(names)}'")
    assert len(names) > 0, ValueError("Number of backbone must be > 0")

    for name in names:
        assert name in NET_DEFAULT_CONFIG.keys(), ValueError(f"Provided backbone is unavailable. Get '{name}'")
        build_result["name"].append(name)
        model_args: Dict = config.Architecture.backbone.pop(f"{name}_args", DotDict({})).get_dict()

        if model_args.get("weights") is None:
            model_args["weights"] = NET_DEFAULT_CONFIG[name]["weights"]

        model: torch.nn.Module = NET_DEFAULT_CONFIG[name]["model"](**model_args)
        model: torch.fx.GraphModule = create_feature_extractor(
            model, NET_DEFAULT_CONFIG[name]["return_node"],
            concrete_args=NET_DEFAULT_CONFIG[name].get("concrete_args")
        )
        model = _freeze_layer(model)
        model.eval()

        if compile_model:
            model.compile()

        out_channels: int = _get_out_channels(model, name)

        if out_proj:
            mlp: torch.nn.Module = MLP(out_channels, **out_proj)
            out_channels: int = mlp.out_channels

            if "out_proj" not in build_result.keys():
                build_result["out_proj"] = torch.nn.ModuleList([mlp])
            else:
                build_result["out_proj"].append(mlp)

        build_result["backbone"].append(model)
        build_result["out_channels"].append(out_channels)
        build_result["reduce"].append(build_reduce(name, config))
    return (
        build_result["backbone"],
        build_result["name"],
        build_result["reduce"],
        build_result.get("out_proj"),
        build_result["out_channels"]
    )


def build_reduce(name: str, config: DotDict) -> functools.partial:
    reduce_config: Dict[str, Any] = config.Architecture.backbone.pop(f"{name}_reduce", DotDict({})).get_dict()
    if name in NET_2D:
        reduce_name = reduce_config.pop("name", DEFAULT_2D_REDUCE)
        assert reduce_name in NET_2D_REDUCE, ValueError(f"Provided reduce method is not supported, Get '{reduce_name}'")
        reduce: torch.nn.Module = NET_2D_REDUCE[reduce_name]
    else:
        reduce_name = reduce_config.get("name", DEFAULT_3D_REDUCE)
        assert reduce_name in NET_3D_REDUCE, ValueError(f"Provided reduce method is not supported, Get '{reduce_name}'")
        reduce: torch.nn.Module = NET_3D_REDUCE[reduce_name]
    reduce: functools.partial = functools.partial(reduce, **reduce_config)
    return reduce
########################################################################################################################


def _get_out_channels(model: torch.nn.Module | torch.fx.GraphModule, name: str) -> int:
    dummy_input = torch.rand(NET_DEFAULT_CONFIG[name]["dummy_input"])
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
