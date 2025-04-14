# Dirty workaround for module import, which violates PEP8: E402
import os
import sys
import copy
import pathlib
import argparse
import warnings
from typing import Dict
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "../../"))

import torch

from AI.src.runner import Tester
from AI.src.modeling import build_model
from AI.src.data import build_dataloader
from AI.src.metrics import MetricWrapper
from AI.src.utils import DotDict, ConfigReader

torch.set_num_threads(64)
torch.set_num_interop_threads(64)

warnings.filterwarnings("once")
warnings.filterwarnings("ignore", category=DeprecationWarning)

DEFAULT_CONFIG_PATH: Dict[str, pathlib.Path] = {
    "linux": pathlib.Path("../../config/single/linux.json"),
    "win32": pathlib.Path("../../config/single/windows.json")
}


def main(args: argparse.Namespace) -> None:
    config: DotDict = ConfigReader(args.config).config
    dataloader = build_dataloader(copy.deepcopy(config), "test")
    model: torch.nn.Module = build_model(copy.deepcopy(config))
    metrics: MetricWrapper = MetricWrapper(copy.deepcopy(config))
    tester = Tester(config, model, metrics, dataloader)
    tester.fit()
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
