from mlflow import MlflowClient
from mlflow.utils.credentials import get_default_host_creds, MlflowHostCreds

from .HttpClient import HttpClient
from .MlflowHttpClient import MlflowHttpClient

__all__ = ["create_http_client"]


def create_http_client(client: MlflowClient, model_name=None) -> HttpClient | MlflowHttpClient:
    """
    Create MLflow HTTP client from MlflowClient.
    If model_name is a Unity Catalog (UC) model, the returned client is UC-enabled.
    """
    from ...common.model_utils import is_unity_catalog_model
    creds: MlflowHostCreds = get_default_host_creds(client.tracking_uri)

    if model_name and is_unity_catalog_model(model_name):
        return HttpClient("api/2.0/mlflow/unity-catalog", creds.host, creds.token)
    else:
        return MlflowHttpClient(creds.host, creds.token)
