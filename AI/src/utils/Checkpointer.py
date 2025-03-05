import os
from typing import Any

import torch


__all__ = ["Checkpointer"]


class Checkpointer(object):
    def __init__(self,
                 save_path: str,
                 ) -> None:
        self.__save_path = save_path
        super(Checkpointer, self).__init__()
