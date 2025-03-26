import mlflow
from mlflow import MlflowClient
from mlflow.entities import Experiment
from mlflow.exceptions import RestException

from .ExportImportException import ExportImportException
from .MlflowTrackingUriTweak import MlflowTrackingUriTweak

__all__ = [
    "get_experiment",
    "set_experiment",
    "download_artifacts"
]


def get_experiment(client: MlflowClient, exp_id_or_name: str) -> Experiment:
    """
    :return: Gets an experiment either by ID or name.
    """
    exp: Experiment = client.get_experiment_by_name(exp_id_or_name)
    if exp is None:
        try:
            exp = client.get_experiment(exp_id_or_name)
        except Exception as ex:
            raise ExportImportException(
                ex,
                f"Cannot find experiment ID or name '{exp_id_or_name}'. Client: {client}'"
            )
    return exp


def set_experiment(client: MlflowClient, exp_name: str, tags=None) -> Experiment:
    """
    :return: Set experiment
    """
    try:
        if not tags:
            tags = {}

        exp_id: str = client.create_experiment(exp_name, tags=tags)
        exp: Experiment = client.get_experiment(exp_id)
        print(f"Created experiment '{exp.name}' with location '{exp.artifact_location}'")
    except RestException as ex:
        if ex.error_code != "RESOURCE_ALREADY_EXISTS":
            raise ExportImportException(ex, f"Cannot create experiment '{exp_name}'")
        exp: Experiment = client.get_experiment_by_name(exp_name)
        print(f"Using existing experiment '{exp.name}' with location '{exp.artifact_location}'")
    return exp


def download_artifacts(client: MlflowClient,
                       download_uri: str,
                       dst_path: str = None,
                       fix: bool = True
                       ) -> str:
    """
    Apparently the tracking_uri argument is not honored for mlflow.artifacts.download_artifacts().
    It seems that tracking_uri is ignored and the global mlflow.get_tracking_uri() is always used.
    If the two happen to be the same, the operation will succeed.
    If not, it fails.
    Issue: Merge pull request #104 from mingyu89/fix-download-artifacts
    """
    if fix:
        with MlflowTrackingUriTweak(client):
            local_path = mlflow.artifacts.download_artifacts(
                artifact_uri=download_uri,
                dst_path=dst_path,
            )
    else:
        local_path = mlflow.artifacts.download_artifacts(
            artifact_uri=download_uri,
            dst_path=dst_path,
            tracking_uri=client.tracking_uri
        )
    return local_path
