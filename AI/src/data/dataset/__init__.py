from .VideoDataset import VideoDataset

# Resourcewarning bug
from .VideoFolderDataset import VideoFolderDataset


DATASETS = {
    "VideoDataset": VideoDataset,
    "VideoFolderDataset": VideoFolderDataset,
}


__all__ = ["DATASETS", "VideoDataset", "VideoFolderDataset"]
