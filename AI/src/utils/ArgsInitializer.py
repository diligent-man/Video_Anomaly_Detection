import os
import pathlib
import commentjson

from typing import Union, Dict
from .DotDict import DotDict


__all__ = ["ArgsInitializer"]


class ArgsInitializer(object):
    def __init__(self, config_path: Union[str, pathlib.Path]) -> None:
        self.__config_path = config_path

    @staticmethod
    def _post_init(args: Dict):
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
        dirs_to_check = ("ckpt", "log")
        architecture_components = [args.architecture[component].name for component in ["backbone", "neck", "head"]]

        print("""##################  Args post-init  #########################""")
        if args.get("output_path") is None:
            if args.get("project_name") is None:
                project_name: str = "nameless_project"
                args.project_name = project_name
            else:
                project_name: str = args.project_name

            print(f"output path: {args.get('output_path')}")
            print(f"project name: {args.get('project_name')} (default: nameless_project)")

            output_path: str = os.path.join(os.getcwd(), "training_results", project_name)
        else:
            output_path: str = args.output_path

        print(f"Final result path: {output_path}\n")

        # Check services in order to create corresponding path
        if args.get("services") is None:
            print("No additional service is specified")
        else:
            for service in args.services:
                if service.get("apply"):
                    dirs_to_check = (*dirs_to_check, service.name)
                else:
                    setattr(service, "apply", False)

        for directory in dirs_to_check:
            # Add path to class attr
            k = f"{directory}_path"
            v = os.path.join(output_path, args.technique, "_".join(architecture_components), directory)
            args[k] = v

            # Create dir if not exists
            if not os.path.isdir(v):
                os.makedirs(v, 0o777, True)
                print(f"Dir for {k} is created.")
            else:
                print(f"Dir for {directory} has already been around and will be overridden.")
        print("""#####################################################################""")

    def __call__(self, *args, **kwargs) -> DotDict:
        with open(file=self.__config_path, mode="r", encoding="UTF-8") as f:
            json_obj: dict = commentjson.loads(f.read())
            args: DotDict = DotDict(json_obj, key_error_handling="warn")
        self._post_init(args)
        return args
