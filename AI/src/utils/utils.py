import os
import yaml
import pathlib
import commentjson

from typing import Union, Dict, Any

__all__ = ["convert_config_json_to_yaml", "load_config"]


def convert_config_json_to_yaml(srcpath: Union[str, pathlib.Path],
                                dstpath: Union[str, pathlib.Path]
                                ) -> None:
    reader = open(srcpath, "r", encoding="utf-8", errors="ignore")
    writer = open(dstpath, "w", encoding="utf-8", errors="ignore")

    config: dict = commentjson.loads(reader.read())
    yaml.safe_dump(config, writer, indent=4, sort_keys=False)

    reader.close()
    writer.close()
    return None


def load_config(fpath: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """
    :param fpath: path to config file. Currently support json
    :return: config dict
    """
    import json
    JSON_EXT = [".json"]
    YAML_EXT = [".yml", ".yaml"]
    SUPPORTED_CONFIG_EXT = [*JSON_EXT, *YAML_EXT]

    _, ext = os.path.splitext(fpath)
    assert ext in SUPPORTED_CONFIG_EXT, "only support yaml, json files for now"

    with open(file=fpath, mode="r", encoding="UTF-8") as f:
        if ext in JSON_EXT:
            # Use commentjson in lieu of json
            config: dict = commentjson.loads(f.read())
        elif ext in YAML_EXT:
            config = yaml.safe_load(f.read())
    return config


def get_save_dir(save_dir: str, name=None, *args, **kwargs) -> str:
    """
    Returns the directory path for saving outputs, derived from arguments or default settings.

    Args:
        args (SimpleNamespace): Namespace object containing configurations such as 'project', 'name', 'task',
            'mode', and 'save_dir'.
        name (str | None): Optional name for the output directory. If not provided, it defaults to 'args.name'
            or the 'args.mode'.

    Returns:
        (pathlib.Path): Directory path where outputs should be saved.
    """
    # project = args.project or (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task
    if name is not None:
        save_dir: str = os.path.join(save_dir, name)

    save_dir = increment_path(save_dir, exist_ok=False, *args, **kwargs)
    return pathlib.Path(save_dir)



def increment_path(path: str, exist_ok=False, sep="", mkdir=False):
    """
    Increments a file or directory path, i.e., runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    If the path exists and `exist_ok` is not True, the path will be incremented by appending a number and `sep` to
    the end of the path. If the path is a file, the file extension will be preserved. If the path is a directory, the
    number will be appended directly to the end of the path. If `mkdir` is set to True, the path will be created as a
    directory if it does not already exist.

    Args:
        path (str | pathlib.Path): Path to increment.
        exist_ok (bool): If True, the path will not be incremented and returned as-is.
        sep (str): Separator to use between the path and the incrementation number.
        mkdir (bool): Create a directory if it does not exist.

    Returns:
        (pathlib.Path): Incremented path.
    """
    path = pathlib.Path(path)
    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        # Method 1
        for n in range(2, 9999):
            p = f"{path}{sep}{n}{suffix}"
            if not os.path.exists(p):
                break
        path = pathlib.Path(p)

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path
