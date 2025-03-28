"""
https://stackoverflow.com/questions/76367473/how-to-resolve-timeout-errors-while-uploading-large-pca-models-using-mlflow
https://github.com/mlflow/mlflow/issues/8539
"""
import os
import sys
import time
import signal

from pathlib import Path
from subprocess import Popen
from argparse import Namespace, ArgumentParser
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), ".."))


from AI.src.utils.mlflow_export_import import copy_run


def main(args: Namespace) -> None:
    if args.username is not None:
        os.environ["MLFLOW_TRACKING_USERNAME"] = args.username

    if args.password is not None:
        os.environ["MLFLOW_TRACKING_PASSWORD"] = args.password

    proc = Popen(
        [
            "mlflow",
            "server",
            "--backend-store-uri",
            f"file://{Path(args.backend_store_uri)}",
        ]
    )
    time.sleep(5)

    copy_run(
        args.run_id,
        args.experiment_name,
        args.src_uri,
        args.dst_uri
    )
    os.kill(proc.pid, signal.SIGTERM)
    return None


if __name__ == "__main__":
    argument_parser: ArgumentParser = ArgumentParser()

    argument_parser.add_argument("--experiment_name", type=str)
    argument_parser.add_argument("--run_id", type=str)
    argument_parser.add_argument("--backend_store_uri", type=str)

    argument_parser.add_argument("--src_uri", type=str, default="http://127.0.0.1:5000")
    argument_parser.add_argument("--dst_uri", type=str)

    # Remote tracking server auth
    argument_parser.add_argument("--username", type=str, default=None)
    argument_parser.add_argument("--password", type=str, default=None)

    parsed_args: Namespace = argument_parser.parse_args()
    main(parsed_args)
