import os
import pathlib

from typing import Union, List

from .DotDict import DotDict
from .utils import load_config


__all__ = ["ConfigPreprocessor"]


class ConfigPreprocessor(object):
    def __init__(self, config_path: Union[str, pathlib.Path]) -> None:
        self.__config_path = config_path
        self.__config: DotDict = DotDict(load_config(self.__config_path), key_error_handling="warn")
        self._post_init()

    def _check_additional_services(self) -> List[str]:
        services = []

        if self.__config.get("Services") is None:
            print("No additional service is specified")
        else:
            for service in self.__config.Services:
                if service.get("apply"):
                    services.append(service.name)
                else:
                    setattr(service, "apply", False)
        return services

    def _create_save_path(self, output_path: Union[str, pathlib.Path]) -> None:
        services: List[str] = self._check_additional_services()
        dir_names = ("ckpt", "log", *services)
        architecture_components = [self.__config.Architecture[component].name for component in ["backbone", "neck", "head"]]

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

    def _resolve_output_path(self) -> pathlib.Path:
        if self.__config.Global.get("output_path") is None:
            if self.__config.Global.get("project_name") is None:
                project_name: str = "nameless_project"
                self.__config.Global.project_name = project_name
            else:
                project_name: str = self.__config.Global.project_name

            print(f"output path: {self.__config.Global.get('output_path')}")
            print(f"project name: {self.__config.Global.get('project_name')} (default: nameless_project)")

            output_path: str = os.path.join(os.getcwd(), "training_results", project_name)
        else:
            output_path: str = self.__config.output_path
        return pathlib.Path(output_path)

    def _post_init(self):
        """
        Check existence of checkpoint and log path
        If not exists, create dir as the following pattern:
            <output_path>/<technique>/<backbone>_<necks>_<head>/ckpt
                                                               /log
                                                               /other_services (if have)
        Default:
            output_path: ./training_results/
            technique: normal
            services: namely tensorboard, etc.
        """
        print("""##################  Config post-init  #########################""")
        output_path: pathlib.Path = self._resolve_output_path()
        print(f"Final output path: {output_path}\n")

        self._create_save_path(output_path)
        print("""#####################################################################""")

    @property
    def config(self):
        return self.__config
