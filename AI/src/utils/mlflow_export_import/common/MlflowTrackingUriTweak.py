import mlflow
from mlflow import MlflowClient

__all__ = ["MlflowTrackingUriTweak"]


class MlflowTrackingUriTweak(object):
    """
    A context manager that handles a bug in several MLflow methods related to downloading artifacts.
    This manifests itself in two places:
        1. mlflow.MlflowClient.create_model_version
            The client's tracking_uri is not honored. Instead create_model_version
            uses mlflow.tracking_uri internally to download run artifacts.
        2. mlflow.artifacts.download_artifacts
            See download_artifacts() below.
    """

    def __init__(self, client: MlflowClient):
        super(MlflowTrackingUriTweak, self).__init__()

        self.client: MlflowClient = client
        self.original_tracking_uri: str = mlflow.get_tracking_uri()
        mlflow.set_tracking_uri(client.tracking_uri)

    def __enter__(self) -> None:
        pass

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        mlflow.set_tracking_uri(self.original_tracking_uri)
