from typing import Dict, Any, List, Tuple, Union

import torch

from .TAM import TAM
from ..MLP import MLP

__all__ = ["TAModel"]


class TAModel(torch.nn.Module):
    def __init__(self,
                 in_proj_args: List[Dict[str, Any]],
                 out_proj_args: Dict[str, Any],
                 TAM_args: Dict[str, Any]
                 ):
        super().__init__()

        self._verify_in_proj_args(in_proj_args)
        self._verify_out_proj_args(in_proj_args, out_proj_args)
        self._verify_TAM_args(TAM_args, in_proj_args)

        self._num_backbones = len(in_proj_args)
        self._in_proj = torch.nn.ModuleList([MLP(**in_proj_args[i]) for i in range(self._num_backbones)])
        self._out_proj = MLP(**out_proj_args)

        # from pprint import pprint as pp
        # pp(TAM_args)
        self._TAM = TAM(**TAM_args)

    @staticmethod
    def _verify_in_proj_args(in_proj_args: Any) -> None:
        print("Verifying in_project args ...", end=" ")
        assert isinstance(in_proj_args, list), ValueError("Args should be a list of dict for corresponding feature extractors")
        assert len(in_proj_args) > 0, ValueError("Num backbones must be > 0")

        embed_dims = []
        for args in in_proj_args:
            assert args.get("output_dim", None), ValueError("Embed dim must be provided")
            embed_dims.append(args["output_dim"])

        assert len(set(embed_dims)) == 1,ValueError("Embed dim of all MLPs must be equivalent")
        print("Done !")

    @staticmethod
    def _verify_out_proj_args(in_proj_args: List[Dict[str, Any]],
                              out_proj_args: Dict[str, Any]
                              ) -> None:
        print("Verifying out_project args ...", end=" ")
        assert out_proj_args.get("input_dim", None) is not None, ValueError("Input dim must be provided and equal to embed_dim")
        assert in_proj_args[0]["output_dim"] == out_proj_args["input_dim"], ValueError("Output dim of 1st MLP must be equivalent to input dim of 2nd MLP")
        print("Done !\n")

    @staticmethod
    def _verify_TAM_args(TAM_args: Dict[str, Any], in_proj_args: List[Dict[str, Any]]) -> None:
        if TAM_args.get("num_backbones", None) is None:
            print("TAM args does not contains num_backbones, init based on len of in_proj_args")
            TAM_args["num_backbones"] = len(in_proj_args)

        if TAM_args.get("embed_dim", None) is None:
            print("TAM args does not contains embed_dim, init based on output_dim of in_proj_args")
            TAM_args["embed_dim"] = in_proj_args[0]["output_dim"]

        if TAM_args.get("num_heads", None) is None:
            print("Num_heads is set to 1 by default")
            TAM_args["num_heads"] = 1

        assert TAM_args["num_heads"] % in_proj_args[0]["output_dim"], ValueError(f"Embed_dim ({in_proj_args[0]['output_dim']}) \
        must be divisible by ({TAM_args['num_heads']})")

    def forward(self, x: List[torch.Tensor],
                return_TAM_states: bool = False
                ) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """
        :param x: Hidden states from feature extractors. Shape list([Num_backbones, batch_size, seg_len, hidden_dim])
        :param return_TAM_states: Hidden states from TAM. Shape [batch, seq_len, embed_dim]
        :return: if return_TAM_states:
                    logits [batch, seq_len, 1] + TAM hidden_states []

                 else:
                    logits/ anomaly score. Shape [batch, seq_len, 1]
        """
        assert len(x) == len(self._in_proj), "Input has more elements than num_backbones"
        x = torch.stack([self._in_proj[i](x[i]) for i in range(self._num_backbones)], dim=0)
        TAM_states = self._TAM(x)
        x = self._out_proj(TAM_states)

        if return_TAM_states:
            return x, TAM_states
        else:
            return x
