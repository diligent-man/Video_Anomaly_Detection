from typing import Dict, Any

import torch

from AI.src.model.CTA.CTAEncoder import CTAEncoder
from AI.src.model.MLP import MLP

__all__ = ["CTAModel"]


class CTAModel(torch.nn.Module):
    def __init__(self,
                 num_backbones: int,
                 in_proj_args: Dict[str, Any],
                 out_proj_args: Dict[str, Any],
                 CTA_encoder_args: Dict[str, Any]
                 ):
        super().__init__()
        self._num_backbones = num_backbones
        self._embed_dim = CTA_encoder_args.get("embed_dim", None)
        self._in_proj = torch.nn.ModuleList([MLP(**in_proj_args) for _ in range(num_backbones)])
        self._encoder = CTAEncoder(num_backbones, **CTA_encoder_args)
        self._out_proj = MLP(**out_proj_args)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Hidden states from feature extractors. Shape [Num_backbones, batch_size, seg_len, hidden_size]
        :return:
        """
        assert x.dim() == 4, "Required dimension is not satisfied"
        num_backbones, _, seq_len, embed_dim = x.size()

        assert num_backbones == self._num_backbones, "Input backbones does not match"

        x = torch.tensor([self._in_proj[i](x[i]).tolist() for i in range(num_backbones)], device=x.device)
        x = self._encoder(x)
        x = self._out_proj(x)
        return x


def main() -> None:
    device = "cuda"
    config = {
        "MODEL_TEACHER_ARGS": {
            "num_backbones": 3,
            "in_proj_args": {
                "input_dim": 1024,
                "hidden_dim": [1024 // 2, 1024 // 2],
                "output_dim": 512,
                "bias": True,
                "dropout": .5,
                "hidden_activation": "LeakyReLU",
                "hidden_activation_args": None,
                "out_activation": None,
                "out_activation_args": None,
                "layer_order": "fc->drop->act",
                "device": None,
                "dtype": None
            },
            "out_proj_args": {
                "input_dim": 512,
                "hidden_dim": [512, 32],
                "output_dim": 1,
                "bias": True,
                "dropout": .5,
                "hidden_activation": "LeakyReLU",
                "hidden_activation_args": None,
                "out_activation": "Sigmoid",
                "out_activation_args": None,
                "layer_order": "fc->drop->act",
                "device": None,
                "dtype": None
            },
            "CTA_encoder_args": {
                # "num_blocks": 1,
                # "num_heads": 1,
                "relative_attention": True,
                "max_relative_position": 32,
                "embed_dim": 512,
            }
        },

        "MODEL_STUDENT_ARGS": {

        },

    }

    num_backbones = 3
    batch_size = 1
    hiddent_dim = 1024
    seq_len = 32

    inps = torch.randn((num_backbones, batch_size, seq_len, hiddent_dim), device=device)
    model = CTAModel(**config["MODEL_TEACHER_ARGS"]).to(device)
    # print(list(model.modules()))

    outs = model(inps)
    print("Output:", outs.shape)
    return None


if __name__ == '__main__':
    main()
