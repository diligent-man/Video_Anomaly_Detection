"""
Exports a run to a directory.
"""
import os
import time
import json
import warnings
import traceback
from typing import Dict, Any, List

import mlflow
from mlflow import MlflowClient
from mlflow.entities import Run, Metric, FileInfo
from mlflow.exceptions import RestException

from ..common.utils import strip_underscores
from ..common.io_utils import write_export_file
from ..common.filesystem import get_filesystem, mk_local_path
from ..common.timestamp_utils import adjust_timestamps, format_seconds


__all__ = ["export_run"]


def export_run(run_id: str,
               output_dir: str,
               client: MlflowClient,
               export_deleted_runs: bool = False,
               ignore_artifacts: bool = False,
               raise_exception: bool = False
               ) -> None:
    """
    :param run_id: Run ID.
    :param output_dir: Output directory.
    :param client: MlflowClient object.
    :param export_deleted_runs: Export deleted runs.
    :param ignore_artifacts: Ignore artifacts from exported run.
    :param raise_exception: Raise an exception instead of just logging error and returning None.

    :return: Run or None if the run was not exported due to export_deleted_runs or errors.
    """
    mlflow.set_tracking_uri(client.tracking_uri)

    start_time = time.time()
    experiment_id: None | str = None

    try:
        run: Run = client.get_run(run_id)
        dst_path = os.path.join(output_dir, "artifacts")

        if run.info.lifecycle_stage == "deleted" and not export_deleted_runs:
            warnings.warn(f"Not exporting run '{run.info.run_id} because its lifecycle_stage is '{run.info.lifecycle_stage}'")
            return None

        msg: str = json.dumps({
            "run_id": run.info.run_id,
            "lifecycle_stage": run.info.lifecycle_stage,
            "experiment_id": run.info.experiment_id
        }, indent=4)

        experiment_id: str = run.info.experiment_id

        tags: Dict[str, Any] = run.data.tags
        tags = dict(sorted(tags.items()))

        info: Dict[str, Any] = strip_underscores(run.info)
        adjust_timestamps(info, ["start_time", "end_time"])

        mlflow_attr = {
            "info": info,
            "params": run.data.params,
            "metrics": _get_metrics_with_steps(client, run),
            "tags": tags,
            "inputs": _inputs_to_dict(run.inputs)
        }

        write_export_file(output_dir, "run.json", __file__, mlflow_attr)
        fs = get_filesystem(".")
        print("Exporting run:\n", msg)

        # copy artifacts
        artifacts: List[FileInfo] = client.list_artifacts(run.info.run_id)

        if ignore_artifacts:
            warnings.warn(f"Not downloading artifacts for run {run.info.run_id}")
        else:
            # Cuz of https://github.com/mlflow/mlflow/issues/2839
            if len(artifacts) > 0:
                fs.mkdirs(dst_path)
                mlflow.artifacts.download_artifacts(
                    run_id=run.info.run_id,
                    dst_path=mk_local_path(dst_path),
                    tracking_uri=client.tracking_uri
                )

        dur: str = format_seconds(time.time()-start_time)
        print(f"Exported in {dur}\n")

    except RestException as e:
        if raise_exception:
            raise e
        err_msg: str = json.dumps({
            "run_id": run_id,
            "experiment_id": experiment_id,
            "RestException": e.json
        }, indent=4)
        warnings.warn(f"Run export failed (1): {err_msg}\n")

    except Exception as e:
        if raise_exception:
            raise e
        err_msg = json.dumps({
            "run_id": run_id,
            "experiment_id": experiment_id,
            "Exception": e
        }, indent=4)
        warnings.warn(f"Run export failed (2): {err_msg}\n")
        traceback.print_exc()
########################################################################################################################


def _get_metrics_with_steps(client: MlflowClient, run: Run) -> Dict[str, List[Dict[str, Any]]]:
    metrics_with_steps: Dict[str, List[Dict[str, Any]]] = {}

    for metric in run.data.metrics.keys():
        metric_history: List[Metric] = client.get_metric_history(run.info.run_id, metric)
        metric_history: List[Dict[str, Any]] = [strip_underscores(metric) for metric in metric_history]

        for i in metric_history:
            del i["key"]

        metrics_with_steps[metric] = metric_history
    return metrics_with_steps


def _inputs_to_dict(inputs):
    def to_dict(ds):
        return {
            "dataset": strip_underscores(ds.dataset),
            "tags": [strip_underscores(tag) for tag in ds.tags]
        }
    return [to_dict(x) for x in inputs.dataset_inputs]
