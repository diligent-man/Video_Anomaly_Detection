from .VideoDataset import VideoDataset
from .VideoFolderDataset import VideoFolderDataset

datasets = {
    "VideoDataset": VideoDataset,
    "VideoFolderDataset": VideoFolderDataset,
}

__all__ = ["datasets"]
