"""
Registered model utilities.
Note: Code logic was removed due to not using
For further info:
    https://github.com/mlflow/mlflow-export-import/blob/master/mlflow_export_import/common/model_utils.py
"""

__all__ = ["is_unity_catalog_model"]


def is_unity_catalog_model(name):
    return len(name.split(".")) == 3
