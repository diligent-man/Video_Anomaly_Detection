import time

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

    __model: Module
    __optimizer: Optimizer
    __scheduler: LRScheduler
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
        self.__callback = CallbackWrapper(self, get_services(config))
        self.__model = model
        self.__optim = optim
        self.__scheduler = scheduler

        self.control: TrainerControl = TrainerControl()
        self.state = TrainerState(
            stateful_callbacks=[cb for cb in [*self.__callback.callback_lst, self.control]
                                if isinstance(cb, ExportableState)]
        )

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

    @property
    def model(self):
        return self.__model

    @property
    def optim(self) -> Optimizer:
        return self.__optim

    @property
    def scheduler(self) -> LRScheduler:
        return self.__scheduler

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
            print("\nSleeping...\n")
            time.sleep(self.__sleep_time)

        self.__callback("on_train_end")
        print("Training finished")
        return None
