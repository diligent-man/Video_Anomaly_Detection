import os
import time
import copy

from collections import defaultdict
from typing import List, Dict, Callable, Any


import torch

from tqdm import tqdm
from torcheval.metrics import Metric


from .forward_strategy import FORWARD_STRATEGIES

from ..data.model import BatchOutput
from ..metrics import MetricWrapper
from ..callbacks import add_callbacks

from ..utils import DotDict, get_amp_cfg, EarlyStopping


__all__ = ["Trainer"]



import inspect


class BatchForwarder(object):
    """
    Forward model on mini-batch manner. This class can be used in train/ val/ test phases
    """
    __epochs: int
    __cur_epoch: int

    def __init__(self, epochs: int, cur_epoch: int) -> None:
        self.__epochs: int = epochs
        self.__cur_epoch: int = cur_epoch


    def __call__(self,
                 phase: str,
                 forward_strategy: str,
                 model: torch.nn.Module,
                 dataloader: torch.utils.data.DataLoader,
                 metrics: MetricWrapper,
                 amp_cfg: Dict[str, Any],
                 optim: torch.optim.Optimizer = None,
                 scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                 grad_scaler: torch.GradScaler = None
                 ) -> BatchOutput:
        """
        Perform 1 epoch runnning with specific phase and selected forward strategy
        """
        assert phase in ("train", "val", "test"), ValueError("Selected phase is invalid")
        assert forward_strategy in FORWARD_STRATEGIES.keys(), ValueError(f"Selected strategy '{forward_strategy}' is not supported")

        forward_callable: Callable = FORWARD_STRATEGIES[forward_strategy]
        kwargs: Dict[str, Any] = {
            "phase": phase, "epochs": self.__epochs, "cur_epoch": self.__cur_epoch,
            "ctx_manager": torch.set_grad_enabled(phase == "train") if phase in ("train", "val") else torch.inference_mode(),
            "model": model, "dataloader": dataloader, "metrics": metrics, "amp_cfg": amp_cfg,
            "grad_scaler": grad_scaler
        }

        if phase == "train":
            model.train()
            kwargs["optim"] = optim

            if "scheduler" in inspect.signature(forward_callable).parameters.keys():
                kwargs["scheduler"] = scheduler
        else:
            model.eval()

        batch_output: BatchOutput = forward_callable(**kwargs)
        return batch_output


class Trainer(object):
    __config: DotDict
    __model: torch.nn.Module
    __lr_scheduler: torch.optim.lr_scheduler.LRScheduler
    __metrics: MetricWrapper
    __train_dataloader: torch.utils.data.DataLoader
    __val_dataloader: torch.utils.data.DataLoader
    __callbacks: Dict[str, List[Callable]] = defaultdict(list, {})

    __start_epoch: int
    __train_batch_forwarder: BatchForwarder
    __test_batch_forwarder: BatchForwarder
    __best_val_loss: float
    __amp_cfg: Dict[str, Any]
    __grad_scaler: torch.GradScaler
    __early_stopping: None | EarlyStopping

    def __init__(self,
                 config: DotDict,
                 model: torch.nn.Module,
                 optim: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler,
                 metrics: MetricWrapper,
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


        # Declare misc attrs used during training
        self.__start_epoch = 0
        self.__best_val_loss = self._get_best_val_loss()
        self.__amp_cfg, self.__grad_scaler = get_amp_cfg(self.__config)
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
        ckpt_path: str = self.__config.Global.ckpt_path

        if self.__config.Checkpoint.get("load", False) and os.path.exists(ckpt_path):
            if "best_ckpt.pt" in os.listdir(ckpt_path):
                return torch.load(f=os.path.join(ckpt_path, "best_checkpoint.pt"))["val_loss"]
        else:
            return float("inf")

    def _setup_train(self):
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

        # Init early stopping
        apply_early_stopping = self.__config.Early_stopping.pop("apply", False)
        if apply_early_stopping:
            self.__early_stopping = EarlyStopping(self.__best_val_loss,
                                                  **self.__config.Early_stopping.get("args", DotDict({})).get_dict()
                                                  )
        else:
            self.__early_stopping = None

    # def _fit(self,
    #          epoch: int,
    #          dataloader: torch.utils.data.DataLoader,
    #          metrics: List[Metric] = None,
    #          phase="train"
    #          ) -> Dict[str, Any]:
    #
    #     return run_epoch_result

    def fit(self):
        """
        As the rule of thumb, model is trained on the mini-batch manner (iteration), which means that after each
        iteration-based forward pass, we do the following:
            a/ compute loss by backprop,
            b/ compute metric
            c/ saving & logging training results and the relevant
            d/ check early stopping cond

        Note: # iters = epochs * len(dataloader)
        """
        print("Start training model ...")
        for epoch in range(self.__start_epoch, self.__start_epoch + self.__config.Global.epochs):
            self.run_callbacks("on_train_epoch_start")

            for phase, dataloader in zip(("train", "val"), (self.__train_dataloader, self.__val_dataloader)):
                batch_output: BatchOutput = BatchForwarder(
                    self.__config.Global.epochs,
                    epoch
                )(
                    phase,
                    self.__config.Data[phase].forward_strategy,
                    self.__model,
                    dataloader,
                    self.__metrics,
                    self.__amp_cfg,
                    self.__optimizer,
                    self.__scheduler,
                    self.__grad_scaler,
                )

                # run_epoch_result: Dict[str, Any] = {**{"Lr": self.__lr_scheduler.get_last_lr().pop()},
                #                                     **self.__run_epoch(phase, epoch, dataloader, metrics)
                #                                     }

                # Add to tensorboad writer
                # if self.__tensorboard:
                #     self.__tensorboard.add_scalar("Learning rate", run_epoch_result["Lr"], epoch)
                #     self.__tensorboard.add_scalars("Loss", {phase: run_epoch_result["loss"]}, epoch)
                #
                #     if self.__config.METRIC_IN_TRAIN:
                #         tag_scalar_dict: Dict[str, Any] = {f"{phase.capitalize()}_{metric}": run_epoch_result[metric] for metric
                #                                            in self.__config.TENSORBOARD_TRACKING_METRIC}
                #         self.__tensorboard.add_scalars("Metric", tag_scalar_dict, epoch)


                # Logging
                # Do sthg here ...

            # Stop program in the meantime
            print("Sleeping...")
            time.sleep(self.__sleep_time)
        print("Training finished")
        return None






