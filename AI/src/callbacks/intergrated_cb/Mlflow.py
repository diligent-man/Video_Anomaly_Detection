"""
MLflow Logging service. Adopted and modified from Ultralytics, Huggingface src code
"""
import os
import pathlib
import warnings
from subprocess import Popen
from typing import Dict, Any


import mlflow
from mlflow import ActiveRun
from mlflow.entities import Experiment


from ...runner import Trainer
from ...utils import make_border
from ..trainer_cb import TrainerCallback
from ...utils.service import ping_server
from ...utils.mlflow_export_import import copy_run


__all__ = ["Mlflow"]


class Mlflow(TrainerCallback):
    """
    A [`TrainerCallback`] that sends the logs to [MLflow](https://www.mlflow.org/). Can be disabled by setting
    environment variable `DISABLE_MLFLOW_INTEGRATION = TRUE`.
    """
    __initialized: bool = False
    __auto_end_run: bool = True
    __proc: Popen = None

    __save_dir: str
    __prev_run_id: str
    __init_server_on_run: bool
    __username: str
    __password: str
    __remote_tracking_uri: str

    __backend_store_uri: str
    __experiment: Experiment
    __run: ActiveRun

    def __init__(self,
                 save_dir: str,
                 prev_run_id: str = None,
                 init_server_on_run: bool = False,
                 *,
                 username: str = None,
                 password: str = None,
                 remote_tracking_uri: str = None
                 ) -> None:
        if username is None:
            username = os.getenv("MLFLOW_TRACKING_USERNAME")
        else:
            os.environ["MLFLOW_TRACKING_USERNAME"] = username

        if password is None:
            password = os.getenv("MLFLOW_TRACKING_PASSWORD")
        else:
            os.environ["MLFLOW_TRACKING_PASSWORD"] = password

        self.__save_dir = save_dir
        self.__prev_run_id = prev_run_id
        self.__init_server_on_run = init_server_on_run
        self.__username = username
        self.__password = password
        self.__remote_tracking_uri = remote_tracking_uri

    def setup(self, instance: Trainer) -> None:
        """
        Set up the optional MLflow integration.
        """
        top, bottom = make_border("Init mlflow service")
        print(top)

        backend_store_uri: str = instance.config.Global.Mlflow_path
        self.__backend_store_uri = backend_store_uri

        experiment_name: str = backend_store_uri.split(os.sep)[-5]  # project_name
        run_name: str = "_".join(backend_store_uri.split(os.sep)[-4: -1])  # technique_mode_experiment_name

        mlflow.set_tracking_uri("file:" + backend_store_uri)
        self.__experiment = mlflow.set_experiment(experiment_name)

        try:
            run = mlflow.get_run(self.__prev_run_id)
            self.__run = mlflow.start_run(
                run.info.run_id,
                None,
                run_name,
                tags=None if instance.config.Mlflow.get("tags", None) is None else instance.config.Mlflow.tags.get_dict(),
                description=instance.config.Mlflow.get("description", None),
                log_system_metrics=instance.config.Mlflow.get("log_system_metrics", True)
            )
        except mlflow.exceptions.MlflowException:
            self.__run = mlflow.start_run(
                None,
                None,
                run_name,
                tags=None if instance.config.Mlflow.get("tags", None) is None else instance.config.Mlflow.tags.get_dict(),
                description=instance.config.Mlflow.get("description", None),
                log_system_metrics=instance.config.Mlflow.get("log_system_metrics", True)
            )

        model_arch: None | str = instance.config.pop("Model_arch", None)
        if model_arch is not None:
            mlflow.log_text(model_arch, "model_arch.txt", self.__run.info.run_id)

        # Log config
        mlflow.log_dict(
            instance.config.get_dict(),
            f"config{pathlib.Path(instance.config.Global.config_path).suffixes[0]}"
        )

        # Log train/ val dataloader
        mlflow.log_text(f"""Train dataloader info:
    {instance.train_dataloader.__repr__()}
        
Val dataloader info:
    {instance.val_dataloader.__repr__()}""", "dataloader_info.txt", self.__run.info.run_id)

        print(f"""Experiment: {self.__experiment.name} - {self.__experiment.experiment_id}
Run: {run_name} - {self.__run.info.run_id}
Backend store uri: {"file:" + backend_store_uri}
Cmd: 'mlflow server --backend-store-uri {"file:" + backend_store_uri}'
** Disabling by setting services.self._ml_flow.apply=false
""")
        print(bottom)
        self.__auto_end_run = True
        self.__initialized = True
        return None

    def on_init_end(self, instance: Trainer) -> None:
        """
        :param instance: Trainer instance
        :return: Initialize local mlflow run
        """
        try:
            ping_server(self.__remote_tracking_uri, auth=(self.__username, self.__password))
            print("Successfully connect to remote tracking server")
        except ConnectionError:
            warnings.warn("Fail to connect to remote tracking server")

        if not self.__initialized:
            self.setup(instance)

        if self.__init_server_on_run:
            self.__proc = Popen(
                [
                    "mlflow",
                    "server",
                    "--backend-store-uri",
                    f"file://{self.__backend_store_uri}",
                    "--artifacts-destination",
                    f"file://{self.__backend_store_uri}"
                ]
            )

    def on_step_end(self, instance: Trainer) -> None:
        batch_output: Dict[str, Any] = instance.state.batch_output.as_metrics()
        mlflow.log_metrics(batch_output, step=instance.state.batch_output.step)

    def on_train_end(self, instance: Trainer) -> None:
        mlflow.log_artifacts(
            instance.config.Global.ckpt_path,
            "ckpt"
        )

        if self.__remote_tracking_uri is not None:
            copy_run(
                self.__run.info.run_id,
                self.__experiment.name,
                mlflow.get_tracking_uri(),
                self.__remote_tracking_uri
            )
        if self.__init_server_on_run:
            self.__proc.kill()

    def __del__(self):
        # if the previous run is not terminated correctly, the fluent API will
        # not let you start a new run before the previous one is killed
        if (
            self.__auto_end_run
            and callable(getattr(mlflow, "active_run", None))
            and mlflow.active_run() is not None
        ):
            mlflow.end_run()
