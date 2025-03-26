"""
Find artifacts of a run that match a name.
"""
import os
import sys
from typing import List

from mlflow import MlflowClient
from mlflow.entities import FileInfo


__all__ = ["find_run_model_names", "find_artifacts"]


def find_run_model_names(client: MlflowClient,
                         run_id: str
                         ) -> List[str]:
    """ 
    Return a list of model artifact directory paths of an MLflow run. 
    Looks for any directory with an 'MLmodel' file and returns its directory.
    """
    matches: List[str] = find_artifacts(client, run_id, "", "MLmodel")
    matches = list(map(lambda match: match.replace("/MLmodel", "").replace("MLmodel", ""), matches))
    return matches


def find_artifacts(client: MlflowClient,
                   run_id: str,
                   path: str,
                   target: str,
                   max_level: int = sys.maxsize
                   ) -> List[str]:
    return _find_artifacts(client, run_id, path, target, max_level, 0, [])


def _find_artifacts(client: MlflowClient,
                    run_id: str,
                    path: str,
                    target: str,
                    max_level: int,
                    level: int,
                    matches: List[str],
                    ) -> List[str]:
    if level + 1 > max_level:
        return matches

    artifacts: List[FileInfo] = client.list_artifacts(run_id, path)
    for artifact in artifacts:
        filename = os.path.basename(artifact.path)

        if filename == target:
            matches.append(artifact.path)

        # NOTE: as of mlflow 2.11.x a new directory 'metadata' is appeared with duplicate MLmodel and friend files in.
        if artifact.is_dir and filename != "metadata":
            _find_artifacts(client, run_id, artifact.path, target, max_level, level+1, matches)
    return matches
