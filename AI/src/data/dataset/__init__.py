from .VideoDataset import VideoDataset
from .VideoFolderDataset import VideoFolderDataset
from .VADVideoLevelDataset import VADVideoLevelDataset
# from .VADFrameLevelDataset import VADFrameLevelDataset


DATASETS = {
    "VideoDataset": VideoDataset,
    "VideoFolderDataset": VideoFolderDataset,
    "VADVideoLevelDataset": VADVideoLevelDataset,
}
# datasets = {
#     "VideoDataset": VideoDataset,
#     "VideoFolderDataset": VideoFolderDataset,
# }
# __all__ = ["datasets", "VideoDataset", "VideoFolderDataset"]


__all__ = ["DATASETS", "VideoDataset", "VideoFolderDataset", "VADVideoLevelDataset"]