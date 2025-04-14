import json

from typing import Dict
from datetime import datetime

from multimethod import multimethod


__all__ = ["Logger"]


class Logger:
    __time: datetime

    def __init__(self, phase: str = "train"):
        """
        phase: "train" || "test"
        """
        self.__time: Dict[str, str] = {f"{phase.capitalize()} at": datetime.now().strftime("%d/%m/%Y %H:%M:%S")}

    @multimethod
    def write(self, file: str, log_info: dict, mode: str = "a") -> None:
        """
        This is used for writing multiple time log from model training
        """
        with open(file=file, mode=mode, encoding="UTF-8", errors="ignore") as f:
            f.write(json.dumps(dict(self.__time, **log_info), indent=4))
            f.write(",\n")
        return None

    @multimethod
    def write(self, file: str, log_info: str, writing_mode: str = "w") -> None:
        """
        This is used for writing one-time log from model testing
        """
        with open(file=file, mode=writing_mode, encoding="UTF-8", errors="ignore") as f:
            f.write(log_info)
        return None
