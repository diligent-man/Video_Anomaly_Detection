from typing import List, Dict, Callable

from ...utils import DotDict
from .InceptionI3D import inception_i3d


__all__ = ["build_backbone"]


backbones: Dict[str, Callable] = {
    "i3d": inception_i3d,
}


def _check_backbone_type(config: DotDict):
    backbone_type: None | str = config.Architecture.backbone.get("type")

    assert backbone_type is not None, ValueError("Backbone type must be specified")
    assert backbone_type in ["single", "multiple"], f"Backbone type must be single/ multiple. Get '{backbone_type}' instead."


def _check_backbone_name(config: DotDict) -> List[str]:
    names: None | str | List[str] = config.Architecture.backbone.get("name")
    assert names is not None, ValueError("At leaset 1 backbone name must be specified")

    if isinstance(names, str):
        names: List[str] = [names]

    for name in names:
        assert name in backbones.keys(),f"Current supported backbone: {backbones.keys()}. Get '{name}' instead."
    return names


def build_backbone(config: DotDict):
    _check_backbone_type(config)
    names = _check_backbone_type(config)
    print(names)




