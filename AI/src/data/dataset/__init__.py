from .VideoDataset import VideoDataset

# Resourcewarning bug
from .VideoFolderDataset import VideoFolderDataset


datasets = {
    "VideoDataset": VideoDataset,
    "VideoFolderDataset": VideoFolderDataset,
}

__all__ = ["datasets", "VideoDataset", "VideoFolderDataset"]
