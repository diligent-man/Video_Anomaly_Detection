import os
from collections import defaultdict
from typing import List, Dict, Callable, Any

import torch
from torcheval.metrics import Metric

from ..utils import DotDict, get_amp_cfg, EarlyStopping
from ..callbacks import add_callbacks


__all__ = ["Trainer"]


class Trainer(object):
    __config: DotDict
    __model: torch.nn.Module
    __lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    __metrics: List[Metric]
    __train_dataloader: torch.utils.data.DataLoader
    __val_dataloader: torch.utils.data.DataLoader
    __callbacks: Dict[str, List[Callable]] = defaultdict(list, {})

    __amp_cfg: Dict[str, Any]
    __grad_scaler: torch.GradScaler

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
        self.__sleep_time = self.__config.Global.get("sleep", 0)

        add_callbacks(self)
        self._setup_train()

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

    def _setup_train(self):
        self.run_callbacks("on_pretrain_routine_start")
        self.__start_epoch: int = 1
        self.__amp_cfg, self.__grad_scaler = get_amp_cfg(self.__config)

        # Load trained checkpoint for continue
        if self.__config.Checkpoint.load:
            ckpt_path = self.__config.Checkpoint.resume_name
            assert os.path.exists(ckpt_path), FileNotFoundError

            checkpoint = torch.load(f=ckpt_path, map_location=self.__config.Global.device)

            self.__start_epoch = checkpoint["epoch"] + 1
            self.__model.load_state_dict(checkpoint["model_state_dict"])
            self.__optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            del checkpoint

        # Early stopping
        self.__early_stopping = EarlyStopping(self.__config.Early_stopping)



    def fit(self):

        # self.run_callbacks("on_pretrain_routine_end")
        print(range(self.__start_epoch, self.__start_epoch+10))