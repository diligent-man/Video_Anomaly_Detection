import dataclasses
from typing import Tuple, Dict, Any

__all__ = ["VideoMetadata"]


@dataclasses.dataclass
class VideoMetadata:
    fps: None | int
    duration: None | int
    resolution: None | Tuple[int, int]

    def __init__(self,
                 fps: None | int,
                 duration: None | int,
                 resolution: None | Tuple[int, int]
                 ) -> None:
        self.fps = fps
        self.duration = duration
        self.resolution = resolution

    def to_dict(self) -> Dict[str, Any]:
        metadata = {
            "fps": self.fps,
            "duration": self.duration,
            "resolution": self.resolution
        }
        return metadata
