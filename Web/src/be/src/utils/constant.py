import os
import dataclasses
import numpy as np

from typing import Dict, Any, Tuple
from dataclasses import asdict


from scipy.signal import find_peaks, savgol_filter


@dataclasses.dataclass
class smooth_signal:
    window_length: int = os.environ.get("WINDOW_LENGTH", 15)
    polyorder: int = os.environ.get("POLYORDER", 6)

    def __post_init__(self) -> None:
        __default_dtypes: Dict[str, Any] = {
            "WINDOW_LENGTH": int,
            "POLYORDER": int,
        }

        for k, v in self.__dict__.items():
            new_val: Any = os.getenv(k.upper(), v)

            if not (new_val is None):
                if new_val.upper() == "NONE":
                    new_val: None = None
                else:
                    new_val: int | float = __default_dtypes[k.upper()](new_val)
            setattr(self, k, new_val)

    @classmethod
    def validate_params(cls, signal_length: int) -> Tuple[int, int]:
        smooth_signal_obj = cls()

        """Validate and adjust parameters based on signal length"""
        if smooth_signal_obj.window_length % 2 == 0:
            smooth_signal_obj.window_length += 1

        # Ensure window length is smaller than signal length
        window_length: int = min(smooth_signal_obj.window_length, signal_length - 1)

        # Ensure polyorder is valid for window length
        polyorder: int = min(smooth_signal_obj.polyorder, window_length - 1)
        return window_length, polyorder

    @classmethod
    def apply(cls, signal: np.ndarray) -> np.ndarray:
        window_length, polyorder = cls.validate_params(len(signal))

        """Apply Savitzky-Golay filter to signal"""
        signal: np.ndarray = np.array(signal)

        # Handle very short signals
        if len(signal) <= polyorder + 2:
            return signal.copy()

        if window_length > polyorder:
            return savgol_filter(signal, window_length, polyorder)
        return signal.copy()


@dataclasses.dataclass
class PeakDetector:
    height: float = os.environ.get("HEIGHT", 0.7)
    threshold: float | None = os.environ.get("THRESHOLD", None)
    distance: int | None = os.environ.get("DISTANCE", None)
    prominence: float = os.environ.get("PROMINENCE", 0.4)
    width: int | None = os.environ.get("WIDTH", None)
    wlen: int | None = os.environ.get("WLEN", None)
    rel_height: float = os.environ.get("REL_HEIGHT", 0.5)
    plateau_size: int | None = os.environ.get("PLATEAU_SIZE", None)

    def __post_init__(self) -> None:
        __default_dtypes: Dict[str, Any] = {
            "HEIGHT": float,
            "THRESHOLD": float,
            "DISTANCE": int,
            "PROMINENCE": float,
            "WIDTH": int,
            "WLEN": int,
            "REL_HEIGHT": float,
            "PLATEAU_SIZE": int
        }

        for k, v in self.__dict__.items():
            new_val: Any = os.getenv(k.upper(), v)

            if not (new_val is None):
                if new_val.upper() == "NONE":
                    new_val: None = None
                else:
                    new_val: int | float = __default_dtypes[k.upper()](new_val)
            setattr(self, k, new_val)

    @classmethod
    def detect(cls, signal: np.ndarray) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        print(cls())
        """Detect peaks in the signal using configured parameters"""
        return find_peaks(signal, **asdict(cls()))
