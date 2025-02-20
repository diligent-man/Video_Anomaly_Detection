from .DefaultDataLoader import DefaultDataLoader
from .VADVideoLevelDataLoader import VADVideoLevelDataLoader


DATALOADERS = {
    "DefaultDataLoader": DefaultDataLoader,
    "VADVideoLevelDataset": VADVideoLevelDataLoader
}


__all = ["DATALOADERS", "DefaultDataLoader", "VADVideoLevelDataset"]
