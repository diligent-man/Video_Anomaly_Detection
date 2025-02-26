from .VideoDataset import VideoDataset
from .VideoFolderDataset import VideoFolderDataset
from .VADVideoLevelDataset import VADVideoLevelDataset
from .VADFrameLevelDataset import VADFrameLevelDataset


DATASETS = {
    "VideoDataset": VideoDataset,
    "VideoFolderDataset": VideoFolderDataset,
    "VADVideoLevelDataset": VADVideoLevelDataset,
    "VADFrameLevelDataset": VADFrameLevelDataset
}

__all__ = [
    "DATASETS",
    "VideoDataset",
    "VideoFolderDataset",
    "VADVideoLevelDataset",
    "VADFrameLevelDataset"
]
