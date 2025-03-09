"""
MLflow Logging service. Adopted and modified from Ultralytics, Hugginface src code
"""
import os
import pathlib

from typing import Dict, Any

from ...runner import Trainer
from ..trainer_cb import TrainerCallback
from ...utils import is_mlflow_available, make_border, ModelArchInspector


__all__ = ["MLflowCallback"]


class MLflowCallback(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [MLflow](https://www.mlflow.org/). Can be disabled by setting
    environment variable `DISABLE_MLFLOW_INTEGRATION = TRUE`.
    """
    def __init__(self):
        if not is_mlflow_available():
            raise RuntimeError("MLflowCallback requires mlflow to be installed. Run `pip install mlflow`.")

        import mlflow
        from mlflow.utils.validation import MAX_PARAM_VAL_LENGTH, MAX_PARAMS_TAGS_PER_BATCH

        self._MAX_PARAM_VAL_LENGTH = MAX_PARAM_VAL_LENGTH
        self._MAX_PARAMS_TAGS_PER_BATCH = MAX_PARAMS_TAGS_PER_BATCH

        self._initialized = False
        self._auto_end_run = False
        self._log_artifacts = False
        self._ml_flow = mlflow

    def setup(self, instance: Trainer) -> None:
        """
        Setup the optional MLflow integration.
        """
        top, bottom = make_border("Init mlflow service")
        print(top)

        uri: str = instance.config.Global.mlflow_path

        # username = instance.config.Mlflow.get("username", None)
        # password = instance.config.Mlflow.get("password", None)
        # auth = (username, password) if username and password else None

        experiment_name: str = uri.split(os.sep)[-5]  # project_name
        run_name: str = "_".join(uri.split(os.sep)[-4: -1])  # technique_mode_experiment_name

        self._ml_flow.set_tracking_uri("file:/" + uri)
        self._ml_flow.set_experiment(experiment_name)

        # old_run = self._ml_flow.get_run("7bdad1ed7a4b4c1fab7f897903bd6cff")

        # active_run = self._ml_flow.start_run(run_name=run_name)
        # for key, value in old_run.data.metrics.items():
        #     self._ml_flow.log_metric(key, value)

        # for key, value in old_run.data.params.items():
        # if key == 'REPLACE_WITH_YOUR_KEY':
        #     new_value = REPLACE_WITH_YOUR_VALUE
        #     self._ml_flow.log_param(key, new_value)
        # else:
        #     self._ml_flow.log_param(key, value)

        # try:
        #     ping_server(uri, auth=auth, timeout=(.5, .5), total=3)
        #     self._ml_flow.set_tracking_uri(uri)
        #     self._ml_flow.set_experiment(experiment_name)
        #     print(f"View at {uri}\n")
        # except (rq.exception.Timeout, self._ml_flow.exceptions.MlflowException) as e:
        #     uri = instance.config.Global.mlflow_path
        #     self._ml_flow.set_tracking_uri(uri)
        #     self._ml_flow.set_experiment(experiment_name)
        #     print(f"{e}.\nTracking uri is set to {ANSIColor().CYAN}mlflow_path{ANSIColor().RESET}\n")
        #     print(f"View at http://localhost:5000 with 'self._ml_flow server --backend-store-uri {uri}'\n")

        active_run = self._ml_flow.active_run() or self._ml_flow.start_run(
            run_id=instance.config.Mlflow.get("run_id", None),
            run_name=run_name,
            tags=None if instance.config.Mlflow.get("tags", None) is None else instance.config.Mlflow.tags.get_dict(),
            description=instance.config.Mlflow.get("description", None),
            log_system_metrics=instance.config.Mlflow.get("log_system_metrics", True)
        )

        model_arch: None | ModelArchInspector = instance.config.pop("Model_arch", None)
        if model_arch is not None:
            self._ml_flow.log_text(str(model_arch()), "model_arch.txt", active_run.info.run_id)

        self._ml_flow.log_dict(instance.config.get_dict(),
                               f"config{pathlib.Path(instance.config.Global.config_path).suffixes[0]}")
        self._ml_flow.log_text(f"""Train dataloader info:
            {instance.train_dataloader.__repr__()}\n
        Val dataloader info:
            {instance.val_dataloader.__repr__()}""", "dataloader_info.txt", active_run.info.run_id)

        self._auto_end_run = True
        self._initialized = True

        print(f"""Experiment name: {experiment_name}
Experiment id: {active_run.info.experiment_id}
Run name: {run_name}
Run id: {active_run.info.run_id}
Uri: {"file:/" + uri}
Command: 'mlflow server --backend-store-uri {"file:/" + uri}'
** Disabling by setting services.self._ml_flow.apply=false
""")
        print(bottom)
        return None

    def on_init_end(self, instance: Trainer) -> None:
        if not self._initialized:
            self.setup(instance)

    def on_step_end(self, instance: Trainer) -> None:
        batch_output: Dict[str, Any] = instance.state.batch_output.as_metrics()
        self._ml_flow.log_metrics(batch_output, step=instance.state.batch_output.step)

    # def on_log(self, args, state, control, logs, model=None, **kwargs):
    #     if not self._initialized:
    #         self.setup(args, state, model)
    #     if state.is_world_process_zero:
    #         metrics = {}
    #         for k, v in logs.items():
    #             if isinstance(v, (int, float)):
    #                 metrics[k] = v
    #             elif isinstance(v, torch.Tensor) and v.numel() == 1:
    #                 metrics[k] = v.item()
    #             else:
    #                 logger.warning(
    #                     f'Trainer is attempting to log a value of "{v}" of type {type(v)} for key "{k}" as a metric. '
    #                     "MLflow's log_metric() only accepts float and int types so we dropped this attribute."
    #                 )
    #
    #         if self._async_log:
    #             self._ml_flow.log_metrics(metrics=metrics, step=state.global_step, synchronous=False)
    #         else:
    #             self._ml_flow.log_metrics(metrics=metrics, step=state.global_step)
    #
    # def on_train_end(self, args, state, control, **kwargs):
    #     if self._initialized and state.is_world_process_zero:
    #         if self._auto_end_run and self._ml_flow.active_run():
    #             self._ml_flow.end_run()
    #
    # def on_save(self, args, state, control, **kwargs):
    #     if self._initialized and state.is_world_process_zero and self._log_artifacts:
    #         ckpt_dir = f"checkpoint-{state.global_step}"
    #         artifact_path = os.path.join(args.output_dir, ckpt_dir)
    #         logger.info(f"Logging checkpoint artifacts in {ckpt_dir}. This may take time.")
    #         self._ml_flow.pyfunc.log_model(
    #             ckpt_dir,
    #             artifacts={"model_path": artifact_path},
    #             python_model=self._ml_flow.pyfunc.PythonModel(),
    #         )

    def __del__(self):
        # if the previous run is not terminated correctly, the fluent API will
        # not let you start a new run before the previous one is killed
        if (
            self._auto_end_run
            and callable(getattr(self._ml_flow, "active_run", None))
            and self._ml_flow.active_run() is not None
        ):
            self._ml_flow.end_run()
