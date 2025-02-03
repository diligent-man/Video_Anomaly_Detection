import pathlib
import argparse
import warnings
from AI.src.utils import ArgsInitializer

warnings.filterwarnings("once")


def main(args: argparse.Namespace) -> None:
    args = ArgsInitializer(args.config_path)()
    return None


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config_path",
                      default=pathlib.Path("./config/teacher.json"),
                      type=str,
                      help="Path to config file"
                      )

    args = args.parse_args()
    main(args)
