from typing import Dict, Any

import torch

from .TAM import TAM
from ..MLP import MLP

__all__ = ["TAModel"]


class TAModel(torch.nn.Module):
    def __init__(self,
                 num_backbones: int,
                 in_proj_args: Dict[str, Any],
                 out_proj_args: Dict[str, Any],
                 CTA_encoder_args: Dict[str, Any]
                 ):
        assert num_backbones > 0, ValueError("Num backbone must be > 0")
        super().__init__()

        self._num_backbones = num_backbones
        self._embed_dim = CTA_encoder_args.get("embed_dim", None)
        self._in_proj = torch.nn.ModuleList([MLP(**in_proj_args) for _ in range(num_backbones)])
        self._TAM = TAM(num_backbones, **CTA_encoder_args)
        self._out_proj = MLP(**out_proj_args)

    def forward(self, x: torch.Tensor, return_hidden_states: bool = False) -> torch.Tensor:
        """
        :param x: Hidden states from feature extractors. Shape [Num_backbones, batch_size, seg_len, hidden_size]
        :param return_hidden_states: Hidden states from TAM. Shape [batch, seq_len, embed_dim]
        :return: logits/ anomaly score with shape [batch, seq_len, prob]
        """
        assert x.dim() == 4, "Required dimension is not satisfied"
        num_backbones, _, seq_len, embed_dim = x.size()

        assert num_backbones == self._num_backbones, "Input backbones does not match"

        x = torch.tensor([self._in_proj[i](x[i]).tolist() for i in range(num_backbones)], device=x.device)
        x = self._TAM(x)
        print(x.shape)
        x = self._out_proj(x)
        return x
