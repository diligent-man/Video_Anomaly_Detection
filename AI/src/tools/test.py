# Dirty workaround for module import, which violates PEP8: E402
import os
import sys
import copy
import argparse
import warnings
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "../../"))

import torch

from AI.src.runner import Tester
from AI.src.modeling import build_model
from AI.src.data import build_dataloader
from AI.src.metrics import MetricWrapper
from AI.src.utils import DotDict, load_config, get_amp_cfg

torch.set_num_threads(64)
torch.set_num_interop_threads(64)

warnings.filterwarnings("once")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(args: argparse.Namespace) -> None:
    config: DotDict = DotDict(load_config(args.config))

    config.Global.log_path = args.log_path
    config.Global.device = args.device
    config.Global.resume_ckpt = args.resume_ckpt

    dataloader = build_dataloader(copy.deepcopy(config), "test")
    model: torch.nn.Module = build_model(copy.deepcopy(config))
    metrics: MetricWrapper = MetricWrapper(copy.deepcopy(config))
    tester = Tester(config, model, metrics, dataloader)
    tester.fit()
    return None


if __name__ == "__main__":
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--config",
                                 type=str,
                                 help="Path to config file"
                                 )

    argument_parser.add_argument("--log_path",
                                 type=str,
                                 help="Path to log test result"
                                 )

    argument_parser.add_argument("--resume_ckpt",
                                 type=str,
                                 help="Path to checkpoint"
                                 )

    argument_parser.add_argument("--device",
                                 default="cpu",
                                 type=str
                                 )

    parsed_args = argument_parser.parse_args()
    main(parsed_args)
