import os
import pathlib
import warnings

from typing import Union, List, Dict, Any, Set

from .DotDict import DotDict
from .utils import load_config


__all__ = ["ConfigReader"]


class ConfigReader(object):
    __defaults: Dict[str, Any] = {
        "save_dir": os.path.join(os.path.dirname(os.getcwd()), "training_results"),
        "project_name": "nameless_project",
        "experiment_name": "exp",
        "technique": "single",
        "mode": "train"
    }

    __expected_vals: Dict[str, Any] = {
        "mode": ["train", "test"],
        "dataset": ["train", "val", "test"],
        "technique": ["single", "distillation"],
        "fields": ["global", "data", "architecture", "optimizer", "metric", "loss", "services"]
    }

    # mode_to_check: Set[str]"mode" = {"train", "test"}
    # dataset_to_check: Set[str] = {"train", "val", "test"}
    # arch_to_check: Dict[str, Set[str]] = {"optional": {"backbone"}, "compulsory": {"neck", "head"}}
    # fields_to_check: Set[str] =

    def __init__(self, config_path: Union[str, pathlib.Path]) -> None:
        self.__config_path = config_path
        self.__config: DotDict = DotDict(load_config(self.__config_path), key_error_handling="warn")

        # Post-init setup
        self._structure_check()
        # self._create_save_dir()

    def _structure_check(self):
        print("##################  Config structure sanity check  #########################")
        mode_to_check: Set[str] = {"train", "test"}
        dataset_to_check: Set[str] = {"train", "val", "test"}
        arch_to_check: Dict[str, Set[str]] = {"optional": {"backbone"}, "compulsory": {"neck", "head"}}
        fields_to_check: Set[str] = {"global", "data", "architecture", "optimizer", "metric", "loss", "services"}

        config: Dict[str, Any] = self.__config.get_dict()

        # Fields check
        for field in config.keys():
            field = field.lower()
            assert field in self.__expected_vals["fields"], \
            ValueError(f"'{field}' field is unexpected. Expect {len(fields_to_check)} fields, including {fields_to_check}")

        # global field check
        for name in ["save_dir", "project_name", "technique", "mode", "experiment_name"]:
            val: str = config["Global"].get(name)

            if val is None:
                self.__config.Global[name] = self.__defaults[name]
                print(f"{name} is not specified, default as:\n\t{self.__defaults[name]}")
            else:
                expected_val = self.__expected_vals.get(name)

                if expected_val is not None:
                    assert val in expected_val, ValueError(f"Expect {expected_val}, but get '{val}' instead")

                print(f"{name}: {val}")
            print()

        # data field check
        dataset: Set[str] = set(config["Data"].keys())
        be = "is" if len(dataset) == 1 else "are"

        for i in dataset:
            assert i in self.__expected_vals["dataset"], ValueError(f"Only {len(dataset_to_check)} states are allowed. Get '{i}' instead.")
        print(f"There {be} configurations for {', '.join(dataset_to_check)} dataset.")

        # architecture check
        arch: Set[str] = set(config["Architecture"].keys())
        assert arch_to_check["compulsory"].issubset(arch), ValueError(f"Obligatory arch is missing. Only has {arch}")

        arch_to_check: List[str] = [*arch_to_check["compulsory"], *arch_to_check["optional"]]

        for i in arch:
            assert i in arch_to_check, ValueError(
                f"Architecture must be in {arch}. Get '{i}' instead.")
        print(f"Model contains {', '.join(arch)} parts")

        # services check
        services: None | List[Dict[str, Any]] = config["Services"]

        if services is None:
            print("No additional services are specified")
        else:
            for service in services:
                assert service.get("name") is not None, ValueError(f"Name to the following config was not specified:\n\t{service}")
        print("############################################################################")

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
        services: List[str] = self._get_services()
        output_path: str = self._resolve_save_dir()

        dir_names = ("ckpt", "log", *services)


        architecture_components = []
        for component in ["backbone", "neck", "head"]:
            component = self.__config.Architecture.get("component")

            if component is not None:
                names: None | str | List[str] = component.get("names")

                if isinstance(names, str):
                    names = [names]

            architecture_components.append("-".join(names))

        architecture_components = [self.__config.Architecture[component].name for component in ["backbone", "neck", "head"] if self.__config.Architecture.get(component) is not None]
        print(architecture_components)

        for directory in dir_names:
            # Add path to class attr
            k: str = f"{directory}_path"
            v: str = os.path.join(output_path, self.__config.Global.technique, "_".join(architecture_components), directory)
            self.__config[k] = v

            # Create dir if not exists
            if not os.path.isdir(v):
                os.makedirs(v, 0o777, True)
                print(f"Dir for {k} is created.")
            else:
                print(f"Dir for {directory} has already been around and will be overridden.")

            print(f"{k}: {v}") if directory == dir_names[-1] else print(f"{k}: {v}\n")
        return None

    def _check_model_architecture(self):
        component_to_check = ["backbone", "neck", "head"]
        for component in component_to_check:
            component_config: None | Dict[str, Any] = self.__config.Architecture.get(component)

    def _resolve_save_dir(self) -> str:
        # <output_path>/ project_name/ < technique > / < mode > / < experiment_name > / ckpt

        path_components: List[str] = ["save_dir", "project_name", "technique", "mode", "experiment_name"]


        if project_name is None:
            project_name: str = "nameless_project"

        if output_path is None:
            output_path: str = os.path.join(os.path.dirname(os.getcwd()), "training_results")


        output_path = os.path.join(output_path, project_name)

        print(f"Output path: {output_path}")
        print(f"Project name: {project_name} (default: nameless_project)")
        return output_path

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
