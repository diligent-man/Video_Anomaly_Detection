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

    def _get_best_val_loss(self) -> float:
        ckpth_path: str = self.__config.Global.ckpt_path

        if self.__config.Checkpoint.get("load", False) and os.path.exists(ckpt_path):
            if "best_checkpoint.pt" in os.listdir(ckpt_path):
                return torch.load(f=os.path.join(ckpth_path, "best_checkpoint.pt"))["val_loss"]
        else:
            return float("inf")

    def _setup_train(self):
        self.__start_epoch: int = 1
        self.__best_val_loss = self._get_best_val_loss()
        self.__amp_cfg, self.__grad_scaler = get_amp_cfg(self.__config)

        self.run_callbacks("on_pretrain_routine_start")

        # Load trained checkpoint for continue
        if self.__config.Checkpoint.load:
            ckpth_path: str = self.__config.Global.ckpt_path
            resume_name: str = self.__config.Checkpoint.get("resume_name", "")

            ckpt_path = os.path.join(ckpth_path, resume_name)
            assert os.path.isfile(ckpt_path), FileNotFoundError

            ckpt = torch.load(f=ckpth_path, map_location="cpu")
            self.__start_epoch = ckpt["epoch"] + 1
            self.__model.load_state_dict(ckpt["model_state_dict"])
            self.__optimizer.load_state_dict(ckpt["optimizer_state_dict"])

            del ckpt

        # Early stopping
        apply_early_stopping = self.__config.Early_stopping.pop("apply", False)
        if apply_early_stopping:
            self.__early_stopping = EarlyStopping(self.__best_val_loss,
                                                  **self.__config.Early_stopping.get("args", DotDict({})).get_dict()
                                                  )
        else:
            self.__early_stopping = None

    def fit(self):
        print(self.__start_epoch)