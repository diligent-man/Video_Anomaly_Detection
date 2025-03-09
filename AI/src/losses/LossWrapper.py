from typing import Union, List, Dict, Any

import torch

from ..utils import DotDict, make_border
from .MILRankingLoss import MILRankingLoss


LOSSES = {
    "L1Loss": torch.nn.L1Loss,
    "NLLLoss": torch.nn.NLLLoss,
    "KLDivLoss": torch.nn.KLDivLoss,
    "MSELoss": torch.nn.MSELoss,
    "BCELoss": torch.nn.BCELoss,
    "BCEWithLogitsLoss": torch.nn.BCEWithLogitsLoss,
    "NLLLoss2d": torch.nn.NLLLoss2d,
    "CosineEmbeddingLoss": torch.nn.CosineEmbeddingLoss,
    "CTCLoss": torch.nn.CTCLoss,
    "HingeEmbeddingLoss": torch.nn.HingeEmbeddingLoss,
    "MarginRankingLoss": torch.nn.MarginRankingLoss,
    "MultiLabelMarginLoss": torch.nn.MultiLabelMarginLoss,
    "MultiLabelSoftMarginLoss": torch.nn.MultiLabelSoftMarginLoss,
    "MultiMarginLoss": torch.nn.MultiMarginLoss,
    "SmoothL1Loss": torch.nn.SmoothL1Loss,
    "HuberLoss": torch.nn.HuberLoss,
    "SoftMarginLoss": torch.nn.SoftMarginLoss,
    "CrossEntropyLoss": torch.nn.CrossEntropyLoss,
    "TripletMarginLoss": torch.nn.TripletMarginLoss,
    "TripletMarginWithDistanceLoss": torch.nn.TripletMarginWithDistanceLoss,
    "PoissonNLLLoss": torch.nn.PoissonNLLLoss,
    "GaussianNLLLoss": torch.nn.GaussianNLLLoss,

    "MILRankingLoss": MILRankingLoss
}


__all__ = ["LossWrapper"]


class LossWrapper:
    __name: str
    __has_aux: bool
    __loss: torch.nn.Module

    def __init__(self, config: DotDict) -> None:
        name: None | str = config.Loss.pop("name", None)
        has_aux: bool = config.Loss.pop("has_aux", False)
        assert name in LOSSES.keys(), ValueError(f"Provided loss is invalid. Get '{name}'")

        self.__name = name
        self.__has_aux = has_aux
        self.__loss = LOSSES[name](**config.Loss.get_dict())

        top, bottom = make_border("Build Loss")
        print(top)
        print(f"""Loss: {name}
Has aux logits: {self.__has_aux}""")
        print(bottom)

    @property
    def name(self) -> str:
        return self.__name

    def _to_float32(self, preds: Union[torch.Tensor, List[torch.Tensor], Dict[str, Any]]
                    ) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
        # cast to float32 in case of using torch.amp.autocast, cuz dtype != float32 can't precisely visualize
        if isinstance(preds, dict):
            for k in preds:
                if isinstance(preds[k], dict) or isinstance(preds[k], list):
                    preds[k] = self._to_float32(preds[k])
                elif isinstance(preds[k], torch.Tensor):
                    preds[k] = preds[k].type(torch.float32)

        elif isinstance(preds, list):
            for k in range(len(preds)):
                if isinstance(preds[k], dict):
                    preds[k] = self._to_float32(preds[k])
                elif isinstance(preds[k], list):
                    preds[k] = self._to_float32(preds[k])
                elif isinstance(preds[k], torch.Tensor):
                    preds[k] = preds[k].type(torch.float32)

        elif isinstance(preds, torch.Tensor):
            preds = preds.type(torch.float32)
        return preds

    def compute_batch_loss(self,
                           inputs: Union[torch.Tensor, List[torch.Tensor]],
                           targets: torch.Tensor = None,
                           aux_logits_weight: float = 0.3
                           ) -> torch.Tensor:
        if isinstance(inputs, torch.Tensor):
            inputs = [inputs]

        inputs = self._to_float32(inputs)

        # aux logits (GoogleLeNet, InceptionV3)
        if self.__has_aux:
            # Test later
            batch_loss = [self.__loss(inputs[i], targets) for i in range(len(inputs))]
            batch_loss = batch_loss[0] + sum(batch_loss) * aux_logits_weight
        else:
            batch_loss = self.__loss(*inputs, targets) if targets is not None else self.__loss(*inputs)
        return batch_loss
