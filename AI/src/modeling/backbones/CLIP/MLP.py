"""
Adopted from huggingface src code. Nothing changed !
"""
import torch

from transformers.activations import ACT2FN


__all__ = ["MLP"]


class MLP(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.activation_fn = ACT2FN[config.hidden_act]
        self.fc1 = torch.nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = torch.nn.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = self.activation_fn(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states
