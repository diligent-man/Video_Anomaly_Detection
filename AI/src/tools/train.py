# Dirty workaround for module import, which violates PEP8: E402
import os
import sys
import copy
import pathlib
import argparse
import warnings
from typing import Dict
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), ".."))


import torch

from AI.src.modeling import build_model
from AI.src.data import build_dataloader
from AI.src.optimizer import build_optimizer

from AI.src.losses import LossWrapper
from AI.src.metrics import MetricWrapper

from AI.src.runner import Trainer
from AI.src.utils import DotDict, ConfigReader

warnings.filterwarnings("once")
warnings.filterwarnings("ignore", category=DeprecationWarning)

DEFAULT_CONFIG_PATH: Dict[str, pathlib.Path] = {
    "linux": pathlib.Path("../../config/single/linux.json"),
    "win32": pathlib.Path("../../config/single/windows.json")
}


def main(args: argparse.Namespace) -> None:
    config: DotDict = ConfigReader(args.config).config

    train_dataloader = build_dataloader(copy.deepcopy(config), "train")
    val_dataloader = build_dataloader(copy.deepcopy(config), "val")

    model: torch.nn.Module = build_model(copy.deepcopy(config))

    optim, scheduler = build_optimizer(copy.deepcopy(config), model)

    loss: LossWrapper = LossWrapper(copy.deepcopy(config))
    metrics: MetricWrapper = MetricWrapper(copy.deepcopy(config))
    trainer = Trainer(config, model, optim, scheduler, loss, metrics, train_dataloader, val_dataloader)
    trainer.fit()
    return None


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--config",
                                 default=DEFAULT_CONFIG_PATH[sys.platform],
                                 type=str,
                                 help="Path to config file"
                                 )

    parsed_args = argument_parser.parse_args()
    main(parsed_args)
