import pathlib
import argparse
import warnings

from AI.src.utils import ConfigPreprocessor
from AI.src.utils.DotDict import DotDict

warnings.filterwarnings("once")


def main(args: argparse.Namespace) -> None:
    # init dist environment
    # if config["Global"]["distributed"]:
    #     dist.init_parallel_env()
    #
    config: DotDict = ConfigPreprocessor(args.config_path).config
    return None


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config_path",
                      default=pathlib.Path("../config/teacher.json"),
                      type=str,
                      help="Path to config file"
                      )

    args = args.parse_args()
    main(args)
