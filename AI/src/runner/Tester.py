import os
import json
import pandas as pd

from typing import List

import torch
from torch import Tensor
from torch.nn import Module
from torch.utils.data import DataLoader


from ..metrics import MetricWrapper
from ..callbacks import CallbackWrapper
from ..utils import DotDict, get_services, Logger

from ..utils.runner_utils import ExportableState
from ..utils.BatchForwarder import BatchForwarder
from ..utils.runner_utils.tester import TesterState


__all__ = ["Tester"]


class Tester(object):
    __config: DotDict
    __logger: Logger
    __metric: MetricWrapper
    __test_dataloader: DataLoader
    __callback: CallbackWrapper

    __model: Module

    def __init__(self,
                 config: DotDict,
                 model: Module,
                 metric: MetricWrapper,
                 dataloader: DataLoader
                 ) -> None:
        from ..callbacks import CallbackWrapper  # tmp fix for cyclic import
        self.__config = config
        self.__model = model
        self.__metric = metric
        self.__test_dataloader = dataloader
        self.__logger = Logger("test")
        self.__callback = CallbackWrapper(self, get_services(config))

        self.state = TesterState(
            stateful_callbacks=[cb for cb in [*self.__callback.callback_lst]
                                if isinstance(cb, ExportableState)]
        )
        self.__callback("on_init_end")

    @property
    def config(self) -> DotDict:
        return self.__config

    @property
    def metric(self) -> MetricWrapper:
        return self.__metric

    @property
    def test_dataloader(self) -> DataLoader:
        return self.__test_dataloader

    @property
    def callback(self) -> CallbackWrapper:
        return self.__callback

    @property
    def model(self):
        return self.__model

    @property
    def logger(self) -> Logger:
        return self.__logger

    def compute_metrics(self, delimiter: str = "; ", chunksize: int = 10 ** 6) -> None:
        """
        Compute metrics from inferred result in log
        """
        for chunk in pd.read_csv(
            os.path.join(self.config.Global.log_path, "pred_result.csv"),
            delimiter=delimiter,
            header=None,
            names=["pred", "label", "idx"],
            chunksize=chunksize,
            engine="python"
        ):
            for _, row in chunk.iterrows():
                pred: str
                label: str

                pred, label, i = row.pred, row.label, row.idx

                pred: List[float] = json.loads(pred)
                label: List[float] = json.loads(label)

                pred: Tensor = torch.tensor(pred, dtype=torch.float16)
                label: Tensor = torch.tensor(label, dtype=torch.uint8)

                self.metric.update(pred, label)
        self.metric.compute()
        result = self.metric.get_result(True)
        self.state.metric_result = result

    def fit(self):
        print(f"""Start running inference on test dataset ...""")
        self.__callback("on_begin")
        BatchForwarder(
            self.__config.Data[self.state.phase].forward_strategy,
            self,
            **{
                "overridden_args": self.__config.Data[self.state.phase].get("overridden_args", DotDict({})).get_dict()
            }
        )()

        print(f"""Start computing metrics from inferred results ...""")
        self.compute_metrics()
        self.__callback("on_end")
        print("Testing finished")
        return None
