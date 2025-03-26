"""
Module to handle importing MLflow run data (params, metrics and tags).
Focus is on data that exceed API limits.
See: https://www.mlflow.org/docs/latest/rest-api.html#request-limits.
"""
import math
from functools import partial
from typing import Any, Dict, List, Callable

from mlflow import MlflowClient
from mlflow.entities import Metric, Param, RunTag
from mlflow.utils.validation import MAX_PARAMS_TAGS_PER_BATCH, MAX_METRICS_PER_BATCH

from ..common import utils
from ..common.source_tags import ExportTags
from ..common.source_tags import mk_source_tags_mlflow_tag, mk_source_tags

__all__ = ["import_run_data"]


def import_run_data(client: MlflowClient,
                    run_dict: Dict[str, Any],
                    run_id: str,
                    import_source_tags: bool,
                    src_user_id: str,
                    use_src_user_id: bool
                    ) -> None:
    _log_params(client, run_dict, run_id, MAX_PARAMS_TAGS_PER_BATCH)
    _log_metrics(client, run_dict, run_id, MAX_METRICS_PER_BATCH)
    _log_tags(
        client,
        run_dict,
        run_id,
        MAX_PARAMS_TAGS_PER_BATCH,
        import_source_tags,
        src_user_id,
        use_src_user_id
    )


########################################################################################################################
def _get_param_data(run_dict: Dict[str, Any]) -> List[Param]:
    return [Param(k, v) for k, v in run_dict["params"].items()]


def _log_param_data(run_id: str, params: List[Param], client: MlflowClient) -> None:
    client.log_batch(run_id, params=params)


def _log_params(client: MlflowClient,
                run_dict: Dict[str, Any],
                run_id: str,
                batch_size: int
                ):
    _log_data(run_dict,
              run_id,
              batch_size,
              _get_param_data,
              partial(_log_param_data, client=client)
              )


########################################################################################################################
def _get_metric_data(run_dict: Dict[str, Any]) -> List[Metric]:
    metrics: List[Metric] = []
    for metric, steps in run_dict["metrics"].items():
        for step in steps:
            metrics.append(Metric(metric, step["value"], step["timestamp"], step["step"]))
    return metrics


def _log_metric_data(run_id: str, metrics: List[Metric], client: MlflowClient) -> None:
    client.log_batch(run_id, metrics=metrics)


def _log_metrics(client: MlflowClient,
                 run_dict: Dict[str, Any],
                 run_id: str,
                 batch_size: int
                 ) -> None:
    _log_data(run_dict,
              run_id,
              batch_size,
              _get_metric_data,
              partial(_log_metric_data, client=client)
              )


########################################################################################################################
def _get_tag_data(run_dict: Dict[str, Any],
                  import_source_tags: bool,
                  src_user_id: bool,
                  use_src_user_id: bool
                  ) -> List[RunTag]:
    tags: Dict[str, str] = run_dict["tags"]

    if import_source_tags:
        source_mlflow_tags = mk_source_tags_mlflow_tag(tags)

        info = run_dict["info"]
        source_info_tags = mk_source_tags(info, f"{ExportTags.PREFIX_RUN_INFO}")

        tags = {**tags, **source_mlflow_tags, **source_info_tags}

    tags: List[RunTag] = [RunTag(k, v) for k, v in tags.items()]
    utils.set_dst_user_id(tags, src_user_id, use_src_user_id)
    return tags


def _log_tag_data(run_id: str, tags: List[RunTag], client: MlflowClient) -> None:
    client.log_batch(run_id, tags=tags)


def _log_tags(client: MlflowClient,
              run_dict: Dict[str, Any],
              run_id: str,
              batch_size: int,
              import_source_tags: bool,
              src_user_id: str,
              use_src_user_id: bool
              ) -> None:
    _log_data(
        run_dict,
        run_id,
        batch_size,
        partial(_get_tag_data,
                import_source_tags=import_source_tags,
                src_user_id=src_user_id,
                use_src_user_id=use_src_user_id
                ),
        partial(_log_tag_data, client=client)
    )


########################################################################################################################
def _log_data(run_dict: Dict[str, Any],
              run_id: str,
              batch_size: int,
              get_data: Callable,
              log_data: Callable
              ) -> None:
    metadata: List[Param | Metric | RunTag] = get_data(run_dict)
    num_batches: int = int(math.ceil(len(metadata) / batch_size))

    for j in range(num_batches):
        start: int = j * batch_size
        end: int = start + batch_size

        batch = metadata[start:end]
        log_data(run_id, batch)
