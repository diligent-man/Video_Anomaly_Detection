import os
import time

from collections import defaultdict
from typing import List, Dict, Callable, Any, Tuple


import torch
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler


from ..losses import LossWrapper
from ..metrics import MetricWrapper
from ..data.model import BatchOutput
from ..callbacks import add_callbacks
from AI.src.utils.BatchForwarder import BatchForwarder
from ..utils import DotDict, get_amp_cfg, EarlyStopping, ModelArchInspector


__all__ = ["Trainer"]


class Trainer(object):
    __config: DotDict
    __model: torch.nn.Module
    __optimizer: Optimizer
    __scheduler: LRScheduler
    __loss: LossWrapper
    __metrics: MetricWrapper
    __train_dataloader: DataLoader
    __val_dataloader: DataLoader
    __callbacks: Dict[str, List[Callable]] = defaultdict(list, {})

    __start_epoch: int
    __best_val_loss: float
    __amp_cfg: Dict[str, Any]
    __grad_scaler: torch.GradScaler
    __sleep_time: float
    __device: str

    __early_stopping: None | EarlyStopping
    __train_batch_forwarder: BatchForwarder
    __test_batch_forwarder: BatchForwarder

    __batch_output: BatchOutput = None

    def __init__(self,
                 config: DotDict,
                 model: torch.nn.Module,
                 optim: Optimizer,
                 scheduler: LRScheduler,
                 loss: LossWrapper,
                 metrics: MetricWrapper,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 ) -> None:
        self.__config = config
        self.__model = model
        self.__optimizer = optim
        self.__scheduler = scheduler
        self.__loss = loss
        self.__metrics = metrics
        self.__train_dataloader = train_dataloader
        self.__val_dataloader = val_dataloader

        # Declare misc attrs used during training
        self.__start_epoch = 1
        self.__best_val_loss = self._get_best_val_loss()
        self.__amp_cfg, self.__grad_scaler = get_amp_cfg(self.__config)
        self.__sleep_time = self.__config.Global.get("sleep", 0)
        self.__device: str = self.__config.Global.get("device", "cpu")
        self.__batch_forwarder: BatchForwarder = BatchForwarder(self.__device)

        add_callbacks(self)
        # self._setup_train()

    @property
    def config(self) -> DotDict:
        return self.__config

    @property
    def model(self) -> torch.nn.Module:
        return self.__model

    @property
    def optimizer(self) -> Optimizer:
        return self.__optimizer

    @property
    def scheduler(self) -> LRScheduler:
        return self.__scheduler

    @property
    def train_dataloader(self) -> DataLoader:
        return self.__train_dataloader

    @property
    def val_dataloader(self) -> DataLoader:
        return self.__val_dataloader

    @property
    def callbacks(self) -> Dict[str, List[Callable]]:
        return self.__callbacks

    @property
    def start_epoch(self) -> int:
        return self.__start_epoch

    @start_epoch.setter
    def start_epoch(self, start_epoch: int) -> None:
        self.__start_epoch = start_epoch

    @property
    def amp_config(self) -> Dict[str, Any]:
        return self.__amp_cfg

    @property
    def batch_output(self) -> BatchOutput:
        return self.__batch_output

    @batch_output.setter
    def batch_output(self, batch_output: BatchOutput):
        self.__batch_output = batch_output

    def run_callbacks(self, event: str, *args, **kwargs) -> None:
        """Run all existing callbacks associated with a particular event."""
        for callback in self.__callbacks.get(event, []):
            callback(self, *args, **kwargs)

    def _get_best_val_loss(self) -> float:
        checkpoint_path: str = self.__config.Global.checkpoint_path

        if self.__config.Checkpoint.get("load", False) and os.path.exists(ckpt_patcheckpoint_path):
            if "best_ckpt.pt" in os.listdir(checkpoint_path):
                return torch.load(f=os.path.join(checkpoint_path, "best_checkpoint.pt"))["val_loss"]
        else:
            return float("inf")

    # def _setup_train(self):
        # Load trained checkpoint for continuous training
        # if self.__config.Checkpoint.load:
        #     checkpoint_path: str = self.__config.Global.checkpoint_path
        #     resume_name: str = self.__config.Checkpoint.get("resume_name", "")
        #
        #     checkpoint_path = os.path.join(checkpoint_path, resume_name)
        #     assert os.path.isfile(checkpoint_path), FileNotFoundError
        #
        #     ckpt = torch.load(f=checkpoint_path, map_location="cpu")
        #     self.__start_epoch = ckpt["epoch"] + 1
        #     self.__model.load_state_dict(ckpt["model"])
        #     self.__optimizer.load_state_dict(ckpt["optimizer"])
        #     del ckpt

        # Init early stopping
        # apply_early_stopping = self.__config.Early_stopping.pop("apply", False)
        # if apply_early_stopping:
        #     self.__early_stopping = EarlyStopping(self.__best_val_loss,
        #                                           **self.__config.Early_stopping.get("args", DotDict({})).get_dict()
        #                                           )
        # else:
        #     self.__early_stopping = None

        # Inspect model architecture
        # inspect_model_arch: bool = self.__config.Global.get("inspect_model_arch", False)
        # dummy_shape: None | Tuple[int, ...] = self.__config.Global.get("dummy_input_shape", None)
        # if inspect_model_arch and dummy_shape is not None:
        #     try:
        #         with torch.amp.autocast(**self.__amp_cfg):
        #             model_arch = ModelArchInspector(
        #                 self.__model,
        #                 self.__config.Global.dummy_input_shape,
        #                 depth=self.__config.Global.get("inspect_depth", 3),
        #                 mode="train",
        #                 verbose=0
        #             )
        #         self.__config["Model_arch"] = model_arch
        #     except Exception as e:
        #         self.__config["Model_arch"] = f"Fail to inspect model architecture due to {e}"

    def fit(self):
        """
        As a rule of thumb, model is trained on the mini-batch manner (iteration), which means that after each
        iteration-based forward pass, we do the following:
            a/ compute loss by backprop,
            b/ compute metric
            c/ saving and logging training results and the relevant
            d/ check early stopping cond

        Note: # iters = epochs * len(DataLoader)
        """
        print("Start training model ...")
        self.run_callbacks("on_train_routine_start")

        # for epoch in range(self.__start_epoch, self.__start_epoch + self.__config.Global.epochs):
        #     self.run_callbacks("on_train_epoch_start")
        #
        #     for phase, dataloader in zip(("train", "val"), (self.__train_dataloader, self.__val_dataloader)):
        #         # if phase == "train" or phase == "val":
        #         #     continue
        #
        #         self.__batch_forwarder(
        #             self.__config.Data[phase].forward_strategy,
        #             self,
        #             phase,
        #             self.__model,
        #             dataloader,
        #             self.__amp_cfg,
        #             self.__loss,
        #             self.__metrics if self.config.Metric[f"in_{phase}"] else None,
        #             self.__optimizer if phase == "train" else None,
        #             self.__scheduler if phase == "train" else None,
        #             self.__grad_scaler,
        #             **{
        #                 "epochs": self.__config.Global.epochs,
        #                 "cur_epoch": epoch,
        #                 "overridden_args": self.__config.Data[phase].get("overridden_args", DotDict({})).get_dict()
        #             }
        #         )
        #
        #     # Stop program in the meantime
        #     print("Sleeping...")
        #     time.sleep(self.__sleep_time)
        print("Training finished")
        return None
