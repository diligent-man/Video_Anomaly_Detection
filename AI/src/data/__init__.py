import torch

from ..utils import DotDict
from .dataset import datasets
from .dataloader import dataloaders


def _post_init_check(dl: torch.utils.data.DataLoader) -> None:
    assert len(dl) > 0, "There must be at least one batch in dataloader"


def build_dataloader(config: DotDict,
                     mode: str
                     ) -> torch.utils.data.DataLoader:
    dataset_name: str = config.Data[mode].dataset.pop("name")
    dataloader_name: str = config.Data[mode].dataloader.pop("name")

    assert mode in ["train", "val", "test"], "Mode should be train/val/test."
    assert dataset_name in datasets, ValueError("Current support dataset {}".format(list(datasets.keys())))
    assert dataloader_name in dataloaders, ValueError("Current support dataloader {}".format(list(dataloaders.keys())))

    ds: torch.utils.data.Dataset = datasets[dataset_name](**config.Data[mode].get_dict("dataset"))
    dl: torch.utils.data.DataLoader = dataloaders[dataloader_name](ds, **config.Data[mode].get_dict("dataloader"))

    _post_init_check(dl)
    print(f"{mode.capitalize()} dataloader: {len(dl)}")
    return dl
