# Dirty workaround for module import, which violates PEP8: E402
import os
import sys

sys.path.append(os.path.join(os.getcwd(), "../../"))

import copy
import pathlib
import argparse
import warnings

from AI.src.utils import ConfigReader
from AI.src.utils.DotDict import DotDict

from AI.src.modeling import build_model
from AI.src.data import build_dataloader

warnings.filterwarnings("once")


def main(args: argparse.Namespace) -> None:
    # init dist environment
    # if config["Global"]["distributed"]:
    #     dist.init_parallel_env()
    #
    config: DotDict = ConfigReader(args.config).config

    train_dataloader = build_dataloader(copy.deepcopy(config), "train")
    val_dataloader = build_dataloader(copy.deepcopy(config), "val")

    model = build_model(copy.deepcopy(config))

    import torch
    with torch.amp.autocast(config.Global.device, torch.float16):
        model(torch.randn(32, 3, 15, 224, 224, device="cuda"))
    return None


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",
                      default=pathlib.Path("../config/teacher.json"),
                      type=str,
                      help="Path to config file"
                      )

    args = args.parse_args()
    main(args)
