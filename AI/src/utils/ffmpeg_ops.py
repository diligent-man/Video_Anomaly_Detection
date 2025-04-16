from typing import Dict, Any
from ..constant import VIDEO_CODECS

__all__ = ["find_video_stream"]


def find_video_stream(streams: Dict[str, Any]) -> str:
    stream: Dict[str, Any]

    for i, stream in enumerate(streams):
        if stream["codec_name"] in VIDEO_CODECS:
            return str(i)
