from .HttpClient import HttpClient

__all__ = ["MlflowHttpClient"]


class MlflowHttpClient(HttpClient):
    """
    MLflow API client: api/2.0
    """
    def __init__(self, host=None, token=None):
        super().__init__("api/2.0/mlflow", host, token)
