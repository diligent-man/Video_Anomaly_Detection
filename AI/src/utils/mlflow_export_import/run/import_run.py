"""
Imports a run from a directory.
"""
import os
import traceback
from typing import Dict, Any, Tuple


import mlflow
from mlflow import MlflowClient
from mlflow.entities import RunStatus, Experiment, LifecycleStage, Run
from mlflow.utils.mlflow_tags import MLFLOW_PARENT_RUN_ID


from .run_utils import update_mlmodel_run_id
from .run_data_importer import import_run_data

from ..client.http import create_http_client
from ..common.filesystem import mk_local_path
from ..common.io_utils import read_file_mlflow
from ..common.mlflow_utils import set_experiment
from ..common.ExportImportException import ExportImportException


__all__ = ["import_run"]


def import_run(src: str,
               exp_name: str,
               import_source_tags: bool = False,
               use_src_user_id: bool = False,
               mlmodel_fix=True,
               client: MlflowClient = None
               ) -> Tuple[Run, str | None]:
    """
    :param src: Directory that contains the exported run.
    :param exp_name: Experiment name to add the run to.
    
    :param import_source_tags: Import source information for MLFlow objects
                               and create tags in destination object.
    :param mlmodel_fix: Add correct run ID in imported MLmodel artifact.
                        Can be expensive for deeply nested artifacts.
    :param use_src_user_id: Set the dst similar to src userID.
    :param client: MlflowClient obj.
    
    :return: Imports a run into the specified experiment.
             The run and its parent run ID if the run is a nested run.
    """
    mlflow.set_tracking_uri(client.tracking_uri)
    # http_client = create_http_client(client)

    src_run_path: str = os.path.join(src, "run.json")
    src_run_dct: Dict[str, Any] = read_file_mlflow(src_run_path)

    print(f"Importing run from '{src}'")
    exp: Experiment = set_experiment(client, exp_name)

    run: Run = client.create_run(exp.experiment_id)
    run_id: str = run.info.run_id

    try:
        # Import params, metrics, tags
        import_run_data(
            client,
            src_run_dct,
            run_id,
            import_source_tags,
            src_run_dct["info"]["user_id"],
            use_src_user_id
        )
        # Import inputs (Temporarily ignore)
        # _import_inputs(http_client, src_run_dct, run_id)

        # Import artifacts
        path = mk_local_path(os.path.join(src, "artifacts"))
        if os.path.exists(path):
            client.log_artifacts(run_id, path)

        if mlmodel_fix:
            update_mlmodel_run_id(client, run_id)

        client.set_terminated(run_id, RunStatus.to_string(RunStatus.FINISHED))
        run = client.get_run(run_id)

        if src_run_dct["info"]["lifecycle_stage"] == LifecycleStage.DELETED:
            client.delete_run(run.info.run_id)
            run = client.get_run(run.info.run_id)
    except Exception as e:
        client.set_terminated(run_id, RunStatus.to_string(RunStatus.FAILED))
        traceback.print_exc()
        raise ExportImportException(e, f"Importing run {run_id} of experiment '{exp.name}' failed")

    print(f"Imported run '{src_run_dct['info']['run_id'][:8]}' "
          f"from exp '{src_run_dct['info']['experiment_id'][:8]}' "
          f"into experiment '{exp.name}'  with id '{exp.experiment_id}'"
          )
    return run, src_run_dct["tags"].get(MLFLOW_PARENT_RUN_ID, None)


def _import_inputs(http_client, src_run_dct, run_id):
    inputs = src_run_dct.get("inputs")
    dct = {"run_id": run_id, "datasets": inputs}
    http_client.post("runs/log-inputs", dct)
