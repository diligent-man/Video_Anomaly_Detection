from .VideoDataset import VideoDataset
from .VideoFolderDataset import VideoFolderDataset
from .VADVideoLevelDataset import VADVideoLevelDataset
from .VADFrameLevelDataset import VADFrameLevelDataset
from .VADFrameLevelTestDataset import VADFrameLevelTestDataset
from .KaggleVADVideoLevelDataset import KaggleVADVideoLevelDataset

DATASETS = {
    "VideoDataset": VideoDataset,
    "VideoFolderDataset": VideoFolderDataset,

    "VADVideoLevelDataset": VADVideoLevelDataset,
    "KaggleVADVideoLevelDataset": KaggleVADVideoLevelDataset,

    "VADFrameLevelDataset": VADFrameLevelDataset,
    "VADFrameLevelTestDataset": VADFrameLevelTestDataset
}

__all__ = [
    "DATASETS",
    "VideoDataset",
    "VideoFolderDataset",

    "VADVideoLevelDataset",
    "KaggleVADVideoLevelDataset",

    "VADFrameLevelDataset",
    "VADFrameLevelTestDataset"
]
