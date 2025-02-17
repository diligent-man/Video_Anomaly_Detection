from collections import defaultdict
from typing import List, Dict, Callable

import torch
from torcheval.metrics import Metric

from ..utils import DotDict, get_amp_cfg, get_services
from ..callbacks import add_callbacks


__all__ = ["Trainer"]
from torchvision.models.vgg import vgg11

class Trainer(object):
    __config: DotDict
    __model: torch.nn.Module
    __lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    __metrics: List[Metric]
    __train_dataloader: torch.utils.data.DataLoader
    __val_dataloader: torch.utils.data.DataLoader
    __callbacks: Dict[str, List[Callable]] = defaultdict(list, {})

    def __init__(self,
                 config: DotDict,
                 model: torch.nn.Module,
                 optim: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 metrics: List[Metric],
                 train_dataloader: torch.utils.data.DataLoader,
                 val_dataloader: torch.utils.data.DataLoader,
                 ) -> None:
        self.__config = config
        self.__model = model
        self.__optimizer = optim
        self.__scheduler = scheduler
        self.__metrics = metrics
        self.__train_dataloader = train_dataloader
        self.__val_dataloader = val_dataloader

        add_callbacks(self)
        # amp_cfg, grad_scaler = get_amp_cfg(config)

    @property
    def callbacks(self) -> Dict[str, List[Callable]]:
        return self.__callbacks

    @property
    def config(self) -> DotDict:
        return self.__config

    def run_callbacks(self, event: str):
        """Run all existing callbacks associated with a particular event."""
        for callback in self.__callbacks.get(event, []):
            callback(self)

    def fit(self):
        self.run_callbacks("on_pretrain_routine_end")
