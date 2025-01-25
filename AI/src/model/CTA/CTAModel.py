from typing import Dict, Any

import torch

from AI.src.model.CTA.CTAEncoder import CTAEncoder
from AI.src.model.MLP import MLP

__all__ = ["CTAModel"]


class CTAModel(torch.nn.Module):
    def __init__(self,
                 first_MLP_args: Dict[str, Any],
                 second_MLP_args: Dict[str, Any],
                 CTA_encoder_args: Dict[str, Any]
                 ):
        self.__MLP1 = MLP(**first_MLP_args)
        self.__MLP2 = MLP(**second_MLP_args)
        self.__CTAEncoder = CTAEncoder(**CTA_encoder_args)


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Hidden states from feature extractors. Shape [Video_segments, Hidden_states]
        :return:
        """