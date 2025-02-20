from typing import Union, List

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

    def compute_batch_loss(self,
                           inputs: Union[torch.Tensor, List[torch.Tensor]],
                           targets: torch.Tensor,
                           aux_logits_weight: float=0.3
                           ) -> torch.Tensor:
        # aux logits (GoogleLeNet, InceptionV3)
        if self.__has_aux:
            batch_loss = [self.__loss(inputs[i], targets) for i in range(len(inputs))]
            batch_loss = batch_loss[0] + sum(batch_loss) * aux_logits_weight
        else:
            batch_loss = self.__loss(inputs, targets)
        return batch_loss
