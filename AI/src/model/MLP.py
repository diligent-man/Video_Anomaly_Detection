import torch


__all__ = ["MLP"]


class MLP(torch.nn.Module):
    """
    Two-layer MLP with ReLU activation.
    """
    __fc1: torch.nn.Module
    __fc2: torch.nn.Module
    __activation: torch.nn.Module

    def __init__(self,
                 input_size: int,
                 output_size: int = 512,
                 activation: torch.nn.Module = torch.nn.ReLU()
                 ):
        super(MLP, self).__init__()
        self.__activation = activation
        self.__fc1 = torch.nn.Linear(input_size, 512)
        self.__fc2 = torch.nn.Linear(512, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__fc1(x)
        x = self.__activation(x)
        x = self.__fc2(x)
        return x
