import sys
import time

from typing import Dict, Any


import torch
from torch.nn import Module
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler


from ..callbacks import CallbackWrapper

from ..losses import LossWrapper
from ..metrics import MetricWrapper

from ..utils.BatchForwarder import BatchForwarder
from ..utils.runner_utils.trainer import TrainerControl, TrainerState
from ..utils import DotDict, get_services, ExportableState


__all__ = ["Trainer"]


class Trainer(object):
    __config: DotDict
    __loss: LossWrapper
    __metric: MetricWrapper
    __train_dataloader: DataLoader
    __val_dataloader: DataLoader
    __callback: CallbackWrapper
    __sleep_time: float

    model: Module
    optimizer: Optimizer
    scheduler: LRScheduler
    control: TrainerControl
    state: TrainerState

    def __init__(self,
                 config: DotDict,
                 model: Module,
                 optim: Optimizer,
                 scheduler: LRScheduler,
                 loss: LossWrapper,
                 metric: MetricWrapper,
                 train_dataloader: DataLoader,
                 val_dataloader: DataLoader,
                 ) -> None:
        self.__config = config
        self.__loss = loss
        self.__metric = metric
        self.__train_dataloader = train_dataloader
        self.__val_dataloader = val_dataloader
        self.__sleep_time = self.__config.Global.get("sleep", 0)

        self.__callback = CallbackWrapper(
            self,
            get_services(config),
            model=model,
            optim=optim,
            scheduler=scheduler,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader
        )

        self.model = model
        self.optim = optim
        self.scheduler = scheduler
        self.control: TrainerControl = TrainerControl()
        self.state = TrainerState(
            stateful_callbacks=[cb for cb in [*self.__callback.callback_lst, self.control]
                                if isinstance(cb, ExportableState)]
        )
        # self.__best_val_loss = self._get_best_val_loss()
        self.control = self.__callback("on_init_end", self.control)

    @property
    def config(self) -> DotDict:
        return self.__config

    @property
    def loss(self) -> LossWrapper:
        return self.__loss

    @property
    def metric(self) -> MetricWrapper:
        return self.__metric

    @property
    def train_dataloader(self) -> DataLoader:
        return self.__train_dataloader

    @property
    def val_dataloader(self) -> DataLoader:
        return self.__val_dataloader

    @property
    def callback(self) -> CallbackWrapper:
        return self.__callback

    # def _get_best_val_loss(self) -> float:
    #     checkpoint_path: str = self.__config.Global.checkpoint_path
    #
    #     if self.__config.Checkpoint.get("load", False) and os.path.exists(checkpoint_path):
    #         if "best_ckpt.pt" in os.listdir(checkpoint_path):
    #             return torch.load(f=os.path.join(checkpoint_path, "best_checkpoint.pt"))["val_loss"]
    #     else:
    #         return float("inf")

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
        step/ iteration-based forward pass, we do the following:
            a/ compute loss by backprop,
            b/ compute metric
            c/ saving and logging training results and the relevant
            d/ check early stopping cond

        Note: # iters/ steps = epochs * len(DataLoader)
        """
        print(f"""Start training model ...""")
        self.__callback("on_train_begin")
        for epoch in range(self.state.epoch, self.state.epoch + self.state.epochs):
            BatchForwarder(
                self.__config.Data[self.state.phase].forward_strategy,
                self,
                **{
                    "overridden_args": self.__config.Data[self.state.phase].get("overridden_args", DotDict({})).get_dict()
                }
            )()

            # Stop program in the meantime
            print("Sleeping...\n")
            time.sleep(self.__sleep_time)
            print(self.control.state())
            print(self.state)
            print()
            print()
        self.__callback("on_train_end")

        print(self.control.state())
        print(self.state)

        print("Training finished")
        return None
