from typing import List

__all__ = ["VIDEO_EXTENSIONS", "VIDEO_CODECS"]

VIDEO_EXTENSIONS: List[str] = [
    "webm", "mkv", "flv", "vob", "ogv",
    "ogg", "rrc", "gifv", "mng", "mov",
    "avi", "qt", "wmv", "yuv", "rm",
    "asf", "amv", "mp4", "m4p", "m4v",
    "mpg", "mp2", "mpeg", "mpe", "mpv",
    "m4v", "svi", "3gp", "3g2", "mxf",
    "roq", "nsv", "flv", "f4v", "f4p",
    "f4a", "f4b", "mod"
]

VIDEO_CODECS: List[str] = ["h264", "mpeg4", "vp9", "mjpeg", "av1"]
