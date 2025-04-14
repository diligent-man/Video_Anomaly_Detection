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

    def fit(self):
        print(f"""Start testing model ...""")
        self.__callback("on_begin")
        BatchForwarder(
            self.__config.Data[self.state.phase].forward_strategy,
            self,
            **{
                "overridden_args": self.__config.Data[self.state.phase].get("overridden_args", DotDict({})).get_dict()
            }
        )()
        self.__callback("on_end")
        print("Testing finished")
        return None
