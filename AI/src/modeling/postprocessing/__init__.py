from typing import Dict, Any

import torch

from ...utils import DotDict
from .PseudoLabelRefiner import PseudoLabelRefiner


__all__ = ["build_postprocessing"]


POSTPROCESSINGS: Dict[str, Any] = {
    "PseudoLabelRefiner": PseudoLabelRefiner
}


def build_postprocessing(config: DotDict) -> None | torch.nn.Module:
    config: DotDict = config.Architecture.get("postprocessing", DotDict({}))
    name: None | str = config.pop("name", None)

    if name is None:
        return None
    else:
        assert name in POSTPROCESSINGS.keys(), ValueError(f"Provided postprocessing is unavailable. Get '{name}'")
        postprocessing = POSTPROCESSINGS[name](**config.get_dict())
        return postprocessing
