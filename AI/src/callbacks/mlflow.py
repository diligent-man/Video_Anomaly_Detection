"""MLflow Logging service. Adopted and modified from Ultralytics src code"""
import os
import pathlib
import warnings
import requests as rq

import mlflow

from ..tools import Trainer
from ..utils import ANSIColor
from ..utils.service import ping_server


def sanitize_dict(x):
    """Sanitize dictionary keys by removing parentheses and converting values to floats."""
    return {k.replace("(", "").replace(")", ""): float(v) for k, v in x.items()}
########################################################################################################################


def on_pretrain_routine_end(instance: Trainer) -> None:
    """
    Log training parameters to MLflow at the end of the pretraining routine.

    This function sets up MLflow logging based on environment variables and trainer arguments.
    It sets the tracking URI, experiment name, and run name, then starts the MLflow run if not already active.
    It finally logs the parameters from the trainer.

    :param: trainer (src.tools.Trainer): The training object with arguments and parameters to log.

    Environment Variables:
        MLFLOW_TRACKING_URI: The URI for MLflow tracking. If not set, defaults to 'runs/mlflow'.
        MLFLOW_EXPERIMENT_NAME: The name of the MLflow experiment. If not set, defaults to trainer.args.project.
        MLFLOW_RUN: The name of the MLflow run. If not set, defaults to trainer.args.name.
        MLFLOW_KEEP_RUN_ACTIVE: Boolean indicating whether to keep the MLflow run active after the end of training.
    """
    uri: str = instance.config.Mlflow.get("tracking_uri", "")

    username = instance.config.Mlflow.get("username", None)
    password = instance.config.Mlflow.get("password", None)
    auth = (username, password) if username and password else None

    mlflow_path: str = instance.config.Global.mlflow_path
    experiment_name: str = mlflow_path.split(os.sep)[-5]  # project_name
    run_name: str = "_".join(mlflow_path.split(os.sep)[-4: -1])  # technique_mode_experiment_name

    try:
        ping_server(uri, auth=auth, timeout=(.5, .5), total=3)
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)
    except (rq.exceptions.Timeout, mlflow.exceptions.MlflowException) as e:
        warnings.warn(f"Get '{e}'.\nTracking uri is set to {ANSIColor().CYAN}mlflow_path{ANSIColor().RESET}")

        uri = instance.config.Global.mlflow_path
        mlflow.set_tracking_uri(uri)
        mlflow.set_experiment(experiment_name)

    active_run = mlflow.active_run() or mlflow.start_run(
        run_name=run_name,
        tags=None if instance.config.Mlflow.get("tags", None) is None else instance.config.Mlflow.tags.get_dict(),
        description=instance.config.Mlflow.get("description", None),
        log_system_metrics=instance.config.Mlflow.get("log_system_metrics", True)
    )

    mlflow.log_dict(instance.config.get_dict(), f"config{pathlib.Path(instance.config.Global.config_path).suffixes[0]}")

    if pathlib.Path(uri).is_dir():
        print(f"View at http://localhost:5000 with 'mlflow server --backend-store-uri {uri}'")
    ValueError(f"WARNING ⚠️ Failed to initialize: \nWARNING ⚠️ Not tracking this run")

    print(f"Logging run_id({active_run.info.run_id}) to {uri}. Disabling by services.mlflow.apply=false in config file\n")
    return None


def on_train_epoch_end(trainer: Trainer):
    """Log training metrics at the end of each train epoch to MLflow."""
    if mlflow:
        mlflow.log_metrics(
            metrics={
                **sanitize_dict(trainer.lr),
                **sanitize_dict(trainer.label_loss_items(trainer.tloss, prefix="train")),
            },
            step=trainer.epoch,
        )


def on_fit_epoch_end(trainer: Trainer):
    """Log training metrics at the end of each fit epoch to MLflow."""
    if mlflow:
        mlflow.log_metrics(metrics=sanitize_dict(trainer.metrics), step=trainer.epoch)


def on_train_end(trainer):
    """Log model artifacts at the end of the training."""
    if not mlflow:
        return
    mlflow.log_artifact(str(trainer.best.parent))  # log save_dir/weights directory with best.pt and last.pt
    for f in trainer.save_dir.glob("*"):  # log all other files in save_dir
        if f.suffix in {".png", ".jpg", ".csv", ".pt", ".yaml"}:
            mlflow.log_artifact(str(f))
    keep_run_active = os.environ.get("MLFLOW_KEEP_RUN_ACTIVE", "False").lower() == "true"
    if keep_run_active:
        print(f"mlflow run still alive, remember to close it using mlflow.end_run()")
    else:
        mlflow.end_run()
        print(f"mlflow run ended")

    print(
        f"results logged to {mlflow.get_tracking_uri()}\ndisable with 'yolo settings mlflow=False'"
    )


callbacks = {
    "on_pretrain_routine_end": on_pretrain_routine_end,
    "on_train_epoch_end": on_train_epoch_end,
    "on_fit_epoch_end": on_fit_epoch_end,
    "on_train_end": on_train_end,
}
