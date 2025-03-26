from typing import Dict, Any

__all__ = [
    "set_dst_user_id",
    "mk_tags_dict",
    "mk_key_value_array_dict",
    "strip_underscores",
    "get_user_id",
    "nested_tags"
]


def set_dst_user_id(tags, user_id, use_src_user_id):
    from mlflow.entities import RunTag
    from mlflow.utils.mlflow_tags import MLFLOW_USER
    user_id = user_id if use_src_user_id else get_user_id()
    tags.append(RunTag(MLFLOW_USER, user_id))


########################################################################################################################
# Tags
def mk_tags_dict(tags_array):
    """
    Transform a list of key/value items to a dict.
    """
    return mk_key_value_array_dict(tags_array, "key", "value")


def mk_key_value_array_dict(kv_array, key_name, value_name):
    """
    Transforms a list of 2 item dicts to a dict.
    Example:  [{'key': 'k1', 'value': 'v1'}, {'key': 'k2', 'value': 'v2'}] ==> {'k1': 'v1', 'k2': 'v2' }
    """
    if kv_array is None:
        return {}
    return {
        x[key_name]: x[value_name] for x in kv_array
    }


########################################################################################################################
# Miscellaneous
def strip_underscores(obj: object) -> Dict[str, Any]:
    return {
        k[1:]: v for (k, v) in obj.__dict__.items()
    }


def get_user_id() -> None | str:
    from mlflow.tracking.context.default_context import DefaultRunContext
    return DefaultRunContext().tags()["mlflow.user"]


def nested_tags(dst_client, run_ids_mapping):
    """
    Set the new parentRunId for new imported child runs.
    """
    for _, v in run_ids_mapping.items():
        src_parent_run_id = v.get("src_parent_run_id", None)
        if src_parent_run_id:
            dst_run_id = v["dst_run_id"]
            dst_parent_run_id = run_ids_mapping[src_parent_run_id]["dst_run_id"]
            dst_client.set_tag(dst_run_id, "mlflow.parentRunId", dst_parent_run_id)
