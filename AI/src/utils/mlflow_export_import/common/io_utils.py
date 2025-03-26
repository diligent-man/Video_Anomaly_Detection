import os
import json
import getpass
import platform
from typing import Dict, Any

import yaml
import mlflow

from ..common.filesystem import mk_local_path
from ..common.source_tags import ExportFields
from ..common.timestamp_utils import ts_now_seconds, ts_now_fmt_utc

export_file_version = "2"


__all__ = ["write_export_file", "write_file", "read_file", "read_file_mlflow"]


def write_export_file(dst: str,
                      file: str,
                      script: str,
                      mlflow_attr: Dict[str, Any],
                      info_attr=None
                      ) -> None:
    """
    :param dst: destination path for writing file
    :param file: filename
    :param script: pathname of the file from which the module was loaded
    :param mlflow_attr: Dict contains things related to mlflow run
    :param info_attr:
    :return: return standard formatted JSON file in dst.
    """
    dst: str = mk_local_path(dst)
    path: str = os.path.join(dst, file)

    info_attr: Dict[str, Dict[str, Any]] = {ExportFields.INFO: info_attr} if info_attr else {}

    mlflow_attr: Dict[str, Dict[str, Any]] = {ExportFields.MLFLOW: mlflow_attr}
    mlflow_attr = {**_mk_system_attr(script), **info_attr, **mlflow_attr}
    os.makedirs(dst, exist_ok=True)
    write_file(path, mlflow_attr)


def write_file(path: str, content: Dict[str, Any], file_type=None):
    """
    Write to JSON, YAML or text file.
    """
    path: str = mk_local_path(path)

    if path.endswith(".json"):
        with open(path, "w", encoding="utf-8") as f:
            f.write(json.dumps(content, indent=4) + "\n")
    elif _is_yaml(path, file_type):
        with open(path, "w", encoding="utf-8") as f:
            yaml.dump(content, f)
    else:
        with open(path, "w") as f:
            content: str = json.dumps(content, indent=4) + "\n"
            f.write(content)


def read_file(path: str, file_type=None) -> Any:
    """
    Read a JSON, YAML or text file.
    """
    with open(mk_local_path(path), "r", encoding="utf-8") as f:
        if path.endswith(".json"):
            return json.loads(f.read())
        elif _is_yaml(path, file_type):
            return yaml.safe_load(f)
        else:
            return f.read()


def read_file_mlflow(path: str) -> Any:
    dct: Any = read_file(path)
    return dct[ExportFields.MLFLOW]
########################################################################################################################


def _is_yaml(path: str, file_type=None) -> bool:
    return any(path.endswith(x) for x in [".yaml", ".yml"]) or file_type in ["yaml", "yml"]


def _mk_system_attr(script) -> Dict[str, Dict[str, Any]]:
    """
    Create system JSON standard containing internal export information.
    """
    sys_attr: Dict[str, Any] = {
        "script": os.path.basename(script),
        "export_file_version": export_file_version,
        "export_time": ts_now_seconds,
        "_export_time": ts_now_fmt_utc,
        "mlflow_version": mlflow.__version__,
        "mlflow_tracking_uri": mlflow.get_tracking_uri(),
        "platform": {
            "python_version": platform.python_version(),
            "system": platform.system(),
            "processor": platform.processor()
        },
        "user": getpass.getuser(),
    }

    dbr = os.environ.get("DATABRICKS_RUNTIME_VERSION", None)
    if dbr:
        dbr_attr: Dict[str, Any] = {
            "databricks": {
                "DATABRICKS_RUNTIME_VERSION": dbr,
            }
        }
        sys_attr = {**sys_attr, **dbr_attr}
    return {ExportFields.SYSTEM: sys_attr}
