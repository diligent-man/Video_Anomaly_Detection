from typing import Dict, Any
from torch.utils.data import Dataset, DataLoader

from ..utils import DotDict
from .dataset import DATASETS
from .dataloader import DATALOADERS
from ..utils.misc import make_border
from .transform import build_transforms


def _post_init_check(dl: DataLoader) -> None:
    assert len(dl) > 0, "There must be at least one batch in DataLoader"


def build_dataloader(config: DotDict,
                     mode: str
                     ) -> DataLoader:
    from pprint import pprint as pp
    pp(config.Data.Train)
    dataset_name: str = config.Data[mode].dataset.pop("name")
    dataloader_name: str = config.Data[mode].dataloader.pop("name")

    assert mode in ["train", "val", "test"], "Mode should be train/val/test."
    assert dataset_name in DATASETS, ValueError("Current support dataset {}".format(list(DATASETS.keys())))
    assert dataloader_name in DATALOADERS, ValueError("Current support DataLoader {}".format(list(DATALOADERS.keys())))

    top, bottom = make_border(f"Build {mode} dataloader")
    print(top)

    # Build transforms & target transforms
    transforms_config: None | DotDict = config.Data[mode].dataset.pop("transforms", None)
    if transforms_config is not None:
        transforms_config: Dict[str, Any] = transforms_config.get_dict()

    target_transforms_config: None | DotDict = config.Data[mode].dataset.pop("target_transforms", None)
    if target_transforms_config is not None:
        target_transforms_config: Dict[str, Any] = target_transforms_config.get_dict()

    ds: Dataset = DATASETS[dataset_name](
        transforms=build_transforms(transforms_config),
        target_transforms=build_transforms(target_transforms_config),
        **config.Data[mode].get_dict("dataset")
    )

    dl: DataLoader = DATALOADERS[dataloader_name](ds, **config.Data[mode].get_dict("dataloader"))

    _post_init_check(dl)
    print(f"{mode.capitalize()} dataLoader len: {len(dl)}")
    print(dl)
    print(bottom)
    return dl

