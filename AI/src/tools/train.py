# Dirty workaround for module import, which violates PEP8: E402
import os
import sys
import torch

sys.path.append(os.path.join(os.getcwd(), "../../"))

import copy
import pathlib
import argparse
import warnings

from .Trainer import BaseTrainer
from AI.src.modeling import build_model
from AI.src.data import build_dataloader
from AI.src.optimizer import build_optimizer
from AI.src.metrics import build_metric
from AI.src.utils import DotDict, ConfigReader, load_ckpt, get_amp_status

warnings.filterwarnings("once")

def main(args: argparse.Namespace) -> None:
    # init dist environment
    # if config["Global"]["distributed"]:
    #     dist.init_parallel_env()
    #
    config: DotDict = ConfigReader(args.config).config

    train_dataloader = build_dataloader(copy.deepcopy(config), "train")
    val_dataloader = build_dataloader(copy.deepcopy(config), "val")

    model: torch.nn.Module = build_model(copy.deepcopy(config))
    optimizer, lr_scheduler = build_optimizer(copy.deepcopy(config), model)
    model, optimizer = load_ckpt(copy.deepcopy(config), model, optimizer)

    # loss_class = build_loss(config["Loss"])
    metrics = build_metric(copy.deepcopy(config))

    autocast_config, grad_scaler = get_amp_status(config)
    trainer = Trainer()
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
