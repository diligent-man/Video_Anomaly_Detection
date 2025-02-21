from .DefaultDataLoader import DefaultDataLoader
from .VADVideoLevelDataLoader import VADVideoLevelDataLoader


DATALOADERS = {
    "DefaultDataLoader": DefaultDataLoader,
    "VADVideoLevelDataLoader": VADVideoLevelDataLoader
}


__all = ["DATALOADERS", "DefaultDataLoader", "VADVideoLevelDataLoader"]
