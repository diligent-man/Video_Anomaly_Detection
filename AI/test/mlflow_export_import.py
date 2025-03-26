import os
import shutil
import tempfile
import mlflow

from subprocess import Popen
from AI.src.utils.mlflow_export_import import copy_run


def save_text(path, text):
    with open(path, "w") as f:
        f.write(text)


def log_artifacts(exp_id: str, run_name: str):
    # Upload artifacts
    with mlflow.start_run(experiment_id=exp_id, run_name=run_name) as run:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path_a = os.path.join(tmp_dir, "a.txt")
            save_text(tmp_path_a, "0")

            tmp_sub_dir = os.path.join(tmp_dir, "dir")
            os.makedirs(tmp_sub_dir)

            tmp_path_b = os.path.join(tmp_sub_dir, "b.txt")
            save_text(tmp_path_b, "1")

            mlflow.log_artifact(tmp_path_a)
            mlflow.log_artifacts(tmp_sub_dir, artifact_path="dir")
            return run.info.run_id


def main() -> None:
    # Supposing that remote tracking server has password enabled
    os.environ["MLFLOW_TRACKING_USERNAME"] = "root"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "Root123!"

    src_exp_name: str = "test_mlflow_export_import"
    src_run_name: str = "run 0"
    src_mlflow_uri: str = "http://localhost:5000"

    dst_mlflow_uri: str = "http://113.22.216.242:5000"

    with tempfile.TemporaryDirectory() as tmp_mlflow_dir:
        Popen(
            [
                "mlflow",
                "server",
                "--backend-store-uri",
                f"file://{tmp_mlflow_dir}/backend",
                "--artifacts-destination",
                f"file://{tmp_mlflow_dir}/mlartifacts"
            ]
        )

        mlflow.set_tracking_uri(src_mlflow_uri)

        if mlflow.get_experiment_by_name(src_exp_name) is None:
            src_exp_id: str = mlflow.create_experiment(src_exp_name)
        else:
            src_exp_id: str = mlflow.get_experiment_by_name(src_exp_name).experiment_id

        run_id = log_artifacts(src_exp_id, src_run_name)

        copy_run(
            run_id,
            src_exp_name,
            src_mlflow_uri,
            dst_mlflow_uri
        )

        Popen(["pkill", "-f", "mlflow"])
        shutil.rmtree(tmp_mlflow_dir)
    return None


if __name__ == '__main__':
    main()
