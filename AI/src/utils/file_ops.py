import os
import yaml
import pathlib
import warnings
import commentjson

from typing import Union, Dict, Any


__all__ = [
    "load_config",
    "convert_config_json_to_yaml",
    "create_increment_path"
]


def load_config(fpath: Union[str, pathlib.Path]) -> Dict[str, Any]:
    """
    :param fpath: path to config file. Currently, support json
    :return: config dict
    """
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


def convert_config_json_to_yaml(src: Union[str, pathlib.Path],
                                dst: Union[str, pathlib.Path]
                                ) -> None:
    reader = open(src, "r", encoding="utf-8", errors="ignore")
    writer = open(dst, "w", encoding="utf-8", errors="ignore")

    config: dict = commentjson.loads(reader.read())
    yaml.safe_dump(config, writer, indent=4, sort_keys=False)

    reader.close()
    writer.close()
    return None


def create_increment_path(path: str,
                          exist_ok: bool = False,
                          sep: str = "",
                          mkdir: bool = False,
                          ) -> pathlib.Path:
    """
    Increments a file or directory path, i.e.,
        runs/exp --> runs/exp{sep}2, runs/exp{sep}3, ... etc.

    Cases:
        1/ path exists and `exist_ok` is not True,
            the path will be incremented by appending a number and `sep` to the end of the path.
        2/ path is a file, the file extension will be preserved.
        3/ path is a directory, the number will be appended directly to the end of the path.

    If `mkdir` is set to True, the path will be created as a directory if it does not already exist.

    Note: This fn is adopted from https://github.com/ultralytics/ultralytics/blob/main/ultralytics/utils/files.py

    :param: path (str | pathlib.Path): Path to increment.
    :param: exist_ok (bool): If True, the path will not be incremented and returned as-is.
    :param: sep (str): Separator to use between the path and the incrementation number.
    :param: mkdir (bool): Create a directory if it does not exist.
    :return (pathlib.Path): Incremented path.
    """
    path: pathlib.Path = pathlib.Path(path)

    if path.exists() and not exist_ok:
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")

        i = 2
        while True:
            dir_name = f"{path}{sep}{i}{suffix}"

            if not os.path.exists(dir_name):
                break
            else:
                i += 1

        path = pathlib.Path(dir_name)
    else:
        warnings.warn("Currently using old save dir for saving results, old data will be overwritten.")

    if mkdir:
        path.mkdir(parents=True, exist_ok=True)
    return path
