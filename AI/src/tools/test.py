# Dirty workaround for module import, which violates PEP8: E402
import os
import sys
import copy
import argparse
import warnings
import multiprocessing
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "../../"))

import torch

from AI.src.runner import Tester
from AI.src.modeling import build_model
from AI.src.data import build_dataloader
from AI.src.metrics import MetricWrapper
from AI.src.utils import DotDict, load_config

torch.set_num_threads(64)
torch.set_num_interop_threads(64)

warnings.filterwarnings("once")
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(args: argparse.Namespace) -> None:
    multiprocessing.set_start_method("spawn")  # for batching infer run

    args.device = "cuda"
    args.log_path = "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_2d/v1/tmp_log/"
    args.config = "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_2d/v1/Mlflow/995263845449942640/d4e6cc59499a4abc90cf6410eb9aef25/artifacts/config.json"
    args.resume_ckpt = "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_2d/v1/Mlflow/995263845449942640/d4e6cc59499a4abc90cf6410eb9aef25/artifacts/ckpt/best_epoch18_step4067.pt"

    config: DotDict = DotDict(load_config(args.config))

    config.Global.log_path = args.log_path
    config.Global.device = args.device
    config.Global.resume_ckpt = args.resume_ckpt

    dataloader = build_dataloader(copy.deepcopy(config), "test")

    model: torch.nn.Module = build_model(copy.deepcopy(config))
    metrics: MetricWrapper = MetricWrapper(copy.deepcopy(config))

    tester = Tester(config, model, metrics, dataloader)
    tester.fit()
    # test.compute_metrics()
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
