import tempfile

import mlflow
from mlflow import MlflowClient

from ..run import import_run, export_run
from ..common.MlflowTrackingUriTweak import MlflowTrackingUriTweak


__all__ = ["copy"]


def copy(src_run_id: str,
         dst_exp_name: str,
         src_uri: str,
         dst_uri: str
         ) -> None:
    """
    :param src_run_id: Source run ID (uniquely identified).
    :param dst_exp_name: Destination experiment name (uniquely identified).
    :param src_uri: Source tracking server URI.
    :param dst_uri: Destination tracking server URI.

    :return: Copy run from src to dst tracking server.
    """
    with tempfile.TemporaryDirectory() as tmp_dir:
        client: MlflowClient = mlflow.MlflowClient(src_uri)
        with MlflowTrackingUriTweak(client):
            export_run(
                src_run_id,
                tmp_dir,
                client
            )

        client: MlflowClient = mlflow.MlflowClient(dst_uri)
        with MlflowTrackingUriTweak(client):
            import_run(
                tmp_dir,
                dst_exp_name,
                False,
                False,
                True,
                client
            )
    return None
