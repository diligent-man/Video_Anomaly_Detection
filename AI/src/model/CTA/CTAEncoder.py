import torch

from AI.src.model.CTA import CTALayer
__all__ = ["CTAEncoder"]


class CTAEncoder(torch.nn.Module):
    def __init__(self,
                 num_hidden_layers: int,

                 ):
        self.__num_hidden_layers = torch.nn.ModuleList([CTALayer() for _ in range(num_hidden_layers)])





