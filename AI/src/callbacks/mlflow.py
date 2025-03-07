"""MLflow Logging service. Adopted and modified from Ultralytics src code"""
import os
import pathlib
from typing import Dict, Any, Callable

import mlflow


from ..tools import Trainer
from ..utils import ANSIColor, make_border, ModelArchInspector

__all__ = ["mlflow_callbacks"]


def on_train_routine_start(instance: Trainer) -> None:
    """
    Log training info to local mlflow folder at the beginning of the training routine.
    At this stage, mlflow logs
        + Model arch (if have)
        + Dataloader info
        + Config file

    :param instance: (Trainer) The training object with arguments and parameters to log.
    """
    top, bottom = make_border("Init mlflow service")
    print(top)
    uri: str = instance.config.Global.mlflow_path

    # username = instance.config.Mlflow.get("username", None)
    # password = instance.config.Mlflow.get("password", None)
    # auth = (username, password) if username and password else None

    experiment_name: str = uri.split(os.sep)[-5]  # project_name
    run_name: str = "_".join(uri.split(os.sep)[-4: -1])  # technique_mode_experiment_name

    mlflow.set_tracking_uri("file:/" + uri)
    mlflow.set_experiment(experiment_name)

    # old_run = mlflow.get_run("7bdad1ed7a4b4c1fab7f897903bd6cff")

    # active_run = mlflow.start_run(run_name=run_name)
    # for key, value in old_run.data.metrics.items():
    #     mlflow.log_metric(key, value)

    # for key, value in old_run.data.params.items():
        # if key == 'REPLACE_WITH_YOUR_KEY':
        #     new_value = REPLACE_WITH_YOUR_VALUE
        #     mlflow.log_param(key, new_value)
        # else:
        #     mlflow.log_param(key, value)

    # try:
    #     ping_server(uri, auth=auth, timeout=(.5, .5), total=3)
    #     mlflow.set_tracking_uri(uri)
    #     mlflow.set_experiment(experiment_name)
    #     print(f"View at {uri}\n")
    # except (rq.exckkkeptions.Timeout, mlflow.exceptions.MlflowException) as e:
    #     uri = instance.config.Global.mlflow_path
    #     mlflow.set_tracking_uri(uri)
    #     mlflow.set_experiment(experiment_name)
    #     print(f"{e}.\nTracking uri is set to {ANSIColor().CYAN}mlflow_path{ANSIColor().RESET}\n")
    #     print(f"View at http://localhost:5000 with 'mlflow server --backend-store-uri {uri}'\n")

    active_run = mlflow.active_run() or mlflow.start_run(
        run_id=instance.config.Mlflow.get("run_id", None),
        run_name=run_name,
        tags=None if instance.config.Mlflow.get("tags", None) is None else instance.config.Mlflow.tags.get_dict(),
        description=instance.config.Mlflow.get("description", None),
        log_system_metrics=instance.config.Mlflow.get("log_system_metrics", True)
    )

    model_arch: None | ModelArchInspector = instance.config.pop("Model_arch", None)
    if model_arch is not None:
        mlflow.log_text(str(model_arch()), "model_arch.txt", active_run.info.run_id)

    mlflow.log_dict(instance.config.get_dict(), f"config{pathlib.Path(instance.config.Global.config_path).suffixes[0]}")
    mlflow.log_text(f"""Train dataloader info:
    {instance.train_dataloader.__repr__()}\n
Val dataloader info:
    {instance.val_dataloader.__repr__()}""","dataloader_info.txt", active_run.info.run_id)

    print(f"""Experiment name: {experiment_name}
Experiment id: {active_run.info.experiment_id}
Run name: {run_name}
Run id: {active_run.info.run_id}
Uri: {uri}
Command: 'mlflow server --backend-store-uri {uri}'
** Disabling by setting services.mlflow.apply=false
""")
    print(bottom)
    return None


def on_train_batch_end(instance: Trainer) -> None:
    batch_output: Dict[str, Any] = instance.batch_output.as_metrics()
    mlflow.log_metrics(batch_output, step=instance.batch_output.step)


def on_val_batch_end(instance: Trainer) -> None:
    on_train_batch_end(instance)

# def on_train_epoch_end(trainer: Trainer):
#     """Log training metrics at the end of each train epoch to MLflow."""
#     if mlflow:
#         mlflow.log_metrics(
#             metrics={
#                 **sanitize_dict(trainer.lr),
#                 **sanitize_dict(trainer.label_loss_items(trainer.tloss, prefix="train")),
#             },
#             step=trainer.epoch,
#         )

#
# def on_fit_epoch_end(trainer: Trainer):
#     """Log training metrics at the end of each fit epoch to MLflow."""
#     if mlflow:
#         mlflow.log_metrics(metrics=sanitize_dict(trainer.metrics), step=trainer.epoch)


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


mlflow_callbacks: Dict[str, Callable] = {
    "on_train_routine_start": on_train_routine_start,
    "on_train_batch_end": on_train_batch_end,

    "on_val_batch_end": on_val_batch_end,
    # "on_train_epoch_end": on_train_epoch_end,
    # "on_fit_epoch_end": on_fit_epoch_end,
    # "on_train_end": on_train_end,
}
