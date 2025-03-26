import time
from typing import Dict, List, Any


__all__ = ["adjust_timestamps", "format_seconds", "timestamps_to_sec", "timestamps_to_milli"]


TS_FORMAT = "%Y-%m-%d %H:%M:%S"
ts_now_seconds = round(time.time())
ts_now_fmt_utc = time.strftime(TS_FORMAT, time.gmtime(ts_now_seconds))


def adjust_timestamps(dct: Dict[str, Any], keys: List[str]) -> Dict[str, Any]:
    """
    Add human-readable keys for millisecond timestamps.
    """
    keys = set(keys)
    for key in keys:
        if key in dct:
            dct[f"_{key}"] = timestamps_to_milli(dct[key])
    return dct


def format_seconds(seconds: float) -> str:
    """
    Format second duration h/m/s format, e.g. '6m 40s' or '40s'.
    """
    minutes, seconds = divmod(seconds, 60)
    minutes = round(minutes)
    if minutes:
        seconds = round(seconds)
        return f"{minutes}m {seconds}s"
    else:
        prec = 2 if seconds < .1 else 1
        seconds = round(seconds, prec)
        return f"{seconds}s"


def timestamps_to_sec(seconds: int, as_utc: bool = True) -> None | str:
    """
    Convert epoch seconds to string format
    """
    if not seconds:
        return None
    if as_utc:
        ts = time.gmtime(seconds)
    else:
        ts = time.localtime(seconds)
    return time.strftime(TS_FORMAT, ts)


def timestamps_to_milli(millis: int, as_utc: bool = True) -> None | str:
    """
    Convert epoch milliseconds to string format
    """
    if not millis:
        return None
    return timestamps_to_sec(round(millis/1000), as_utc)
