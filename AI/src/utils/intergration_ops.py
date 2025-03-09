import os

from importlib.util import find_spec


__all__ = ["is_mlflow_available"]


def is_mlflow_available() -> bool:
    if os.getenv("DISABLE_MLFLOW_INTEGRATION", "FALSE").upper() == "TRUE":
        return False
    return find_spec("mlflow") is not None
