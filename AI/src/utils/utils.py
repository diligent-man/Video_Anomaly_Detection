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
