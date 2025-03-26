import os
import tempfile
from typing import List, Any, Dict

from mlflow import MlflowClient

from ..common.io_utils import read_file, write_file
from ..common.mlflow_utils import download_artifacts
from ..common.find_artifacts import find_run_model_names


def update_mlmodel_run_id(client: MlflowClient, run_id: str) -> None:
    """
    :param: mlflow client
    :param: run_id
    :return:
        Workaround to fix the run_id in the dst MLmodel file since there is no method to get all model artifacts of a run.
    Since an MLflow run does not keep track of its models, there is no method to retrieve the artifact path to all its models.
    This workaround recursively searches the run's root artifact directory for all MLmodel files, and assumes their directory
    represents a path to the model.
    """
    mlmodel_paths: List[str] = find_run_model_names(client, run_id)

    for model_path in mlmodel_paths:
        download_uri: str = f"runs:/{run_id}/{model_path}/MLmodel"
        local_path: str = download_artifacts(client, download_uri)
        mlmodel: Dict[str, Any] = read_file(local_path, "yaml")
        mlmodel["run_id"] = run_id

        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = os.path.join(tmp_dir, "MLmodel")
            write_file(output_path, mlmodel, "yaml")

            if model_path == "MLmodel":
                model_path = ""

            client.log_artifact(run_id, output_path, model_path)
