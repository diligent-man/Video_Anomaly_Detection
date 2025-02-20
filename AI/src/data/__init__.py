from torch.utils.data import Dataset, DataLoader

from ..utils import DotDict
from .dataset import DATASETS
from .dataloader import DATALOADERS


def _post_init_check(dl: DataLoader) -> None:
    assert len(dl) > 0, "There must be at least one batch in DataLoader"


def build_dataloader(config: DotDict,
                     mode: str
                     ) -> DataLoader:
    dataset_name: str = config.Data[mode].dataset.pop("name")
    dataloader_name: str = config.Data[mode].dataloader.pop("name")

    assert mode in ["train", "val", "test"], "Mode should be train/val/test."
    assert dataset_name in DATASETS, ValueError("Current support dataset {}".format(list(DATASETS.keys())))
    assert dataloader_name in DATALOADERS, ValueError("Current support DataLoader {}".format(list(DATALOADERS.keys())))

    ds: Dataset = DATASETS[dataset_name](**config.Data[mode].get_dict("dataset"))
    dl: DataLoader = DATALOADERS[dataloader_name](ds, **config.Data[mode].get_dict("dataloader"))

    _post_init_check(dl)
    print(f"{mode.capitalize()} DataLoader: {len(dl)}")
    return dl
