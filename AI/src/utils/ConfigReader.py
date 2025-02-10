import os
import pathlib

from typing import Union, List, Dict, Any, Set
from .DotDict import DotDict
from .utils import load_config, create_increment_path
from . import ANSIColor

__all__ = ["ConfigReader"]


class ConfigReader(object):
    __defaults: Dict[str, Any] = {
        "save_dir": os.path.join(os.path.dirname(os.getcwd()), "results"),
        "project_name": "nameless_project",
        "experiment_name": "run",
        "technique": "single",
        "mode": "train"
    }

    __expected_vals: Dict[str, Any] = {
        "mode": ["train", "test"],
        "dataset": ["train", "val", "test"],
        "technique": ["single", "distillation"],
        "fields": ["global", "data", "architecture", "optimizer", "metric", "loss", "services"]
    }

    def __init__(self, config_path: Union[str, pathlib.Path]) -> None:
        self.__config_path = config_path
        self.__config: DotDict = DotDict(load_config(self.__config_path), key_error_handling="warn")

        # Post-init setup
        print("##################  Config post-init running  ######################### ")
        self._structure_check()
        self._create_save_dir()
        print("############################################################################")

    def _structure_check(self):
        # TODO: Optimize checking process
        print("Config structure sanity check")
        config: Dict[str, Any] = self.__config.get_dict()
        arch_to_check: Dict[str, Set[str]] = {"optional": {"backbone"}, "compulsory": {"neck", "head"}}

        # Fields check
        for field in config.keys():
            field = field.lower()
            assert field in self.__expected_vals["fields"], \
                ValueError(f"'{field}' field is unexpected. Expect {len(self.__expected_vals['fields'])} fields\
                 ({', '.join(self.__expected_vals['fields'])})")

        # global field check
        print("Global:")
        for name in ["save_dir", "project_name", "technique", "mode", "experiment_name"]:
            val: str = config["Global"].get(name)

            if val is None:
                self.__config.Global[name] = self.__defaults[name]
                print(f"\t{ANSIColor().CYAN}{name}{ANSIColor().RESET} is not specified, default as:\n\t\t{self.__defaults[name]}")
            else:
                expected_val = self.__expected_vals.get(name)

                if expected_val is not None:
                    assert val in expected_val, ValueError(f"Expect {expected_val}, but get '{val}' instead")

                print(f"\t{ANSIColor().CYAN}{name}{ANSIColor().RESET}:\n\t\t{val}")
        print()

        # data field check
        dataset: Set[str] = set(config["Data"].keys())

        print(f"Dataset config:")
        for i in dataset:
            assert i in self.__expected_vals["dataset"], ValueError(f"Only {len(self.__expected_vals['dataset'])} states ({', '.join(self.__expected_vals['dataset'])}) are allowed. Get '{i}' instead.")
            assert config["Data"][i].get("dataset") is not None, ValueError(f"Dataset config for {i} set is None")
            assert config["Data"][i].get("dataloader") is not None, ValueError(f"Dataloader config for {i} set is None")
            print(f"\t{ANSIColor().CYAN}{i}{ANSIColor().RESET}: {config['Data'][i]['dataset'].get('name')}")
        print()

        # architecture check
        arch: Set[str] = set(config["Architecture"].keys())
        assert arch_to_check["compulsory"].issubset(arch), ValueError(f"Obligatory arch is missing. Only has {arch}")

        arch_to_check: List[str] = [*arch_to_check["compulsory"], *arch_to_check["optional"]]

        print(f"Model config")
        for i in arch:
            assert i in arch_to_check, ValueError(f"Architecture must be in {arch}. Get '{i}' instead.")
            assert config["Architecture"][i].get("name") is not None, ValueError(f"{i} name is None")
            print(f"\t{ANSIColor().CYAN}{i}{ANSIColor().RESET}: {config['Architecture'][i].get('name')}")
        print()

        # services check
        services: None | List[Dict[str, Any]] = config["Services"]

        if services is None:
            print("No additional services are specified")
        else:
            print("Services:")
            for service in services:
                assert service.get("name") is not None, ValueError(f"Name to the following config was not specified:\n\t{service}")
                print(f"\t{ANSIColor().CYAN}{service['name']}{ANSIColor().RESET}")
        print()

    def _create_save_dir(self) -> None:
        """
        Check existence of checkpoint and log path
        If not exists, create dir as the following pattern:
            <output_path>/<project_name>/<technique>/<mode>/<experiment_name>/ckpt
                                                                             /log
                                                                             /other_services (if have)
        Default:
            output_path: ./training_results/
            technique: normal
            services: namely tensorboard, etc.
        """
        save_dir: List[str] = [self.__config.Global.save_dir,
                               self.__config.Global.project_name,
                               self.__config.Global.technique,
                               self.__config.Global.mode,
                               self.__config.Global.experiment_name
                               ]
        save_dir: str = f"{os.sep}".join(save_dir)
        save_dir: pathlib.Path = create_increment_path(save_dir, False, "", True)

        services: List[str] = self._get_services()
        dirs = ("ckpt", "log", *services)

        print("Save dir:")
        for dir_name in dirs:
            # Add path to class attr
            k: str = f"{dir_name}_path"
            v: str = os.path.join(save_dir, dir_name)

            self.__config[k] = v
            os.makedirs(v, exist_ok=True)

            print(f"{ANSIColor().CYAN}\t{k}{ANSIColor().RESET}: \n\t\t{v}") if dir_name != dirs[-1] else print(f"\t{ANSIColor().CYAN}{k}{ANSIColor().RESET}: \n\t\t{v}\n")
        return None

    def _get_services(self) -> List[str]:
        services: List[str] = []
        service_config: List[Dict[str, Any]] = self.__config.get("Services")

        if service_config is None:
            print("No additional service is specified")
        else:
            for service in service_config:
                apply_status = service.get("apply", False)

                if apply_status:
                    services.append(service.name)
                else:
                    setattr(service, "apply", False)
        return services

    @property
    def config(self):
        return self.__config
