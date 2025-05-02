from .VideoDataset import VideoDataset
from .VideoFolderDataset import VideoFolderDataset
from .VADVideoLevelDataset import VADVideoLevelDataset
from .VADFrameLevelDataset import VADFrameLevelDataset
from .VADFrameLevelTestDataset import VADFrameLevelTestDataset


DATASETS = {
    "VideoDataset": VideoDataset,
    "VideoFolderDataset": VideoFolderDataset,
    "VADVideoLevelDataset": VADVideoLevelDataset,
    "VADFrameLevelDataset": VADFrameLevelDataset,
    "VADFrameLevelTestDataset": VADFrameLevelTestDataset
}

__all__ = [
    "DATASETS",
    "VideoDataset",
    "VideoFolderDataset",
    "VADVideoLevelDataset",
    "VADFrameLevelDataset",
    "VADFrameLevelTestDataset"
]
