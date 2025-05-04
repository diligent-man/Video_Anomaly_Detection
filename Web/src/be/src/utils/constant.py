import dataclasses
import os
import numpy as np
from typing import Dict, Any, Tuple
from scipy.signal import find_peaks, peak_widths, savgol_filter


@dataclasses.dataclass
class smooth_signal:
    window_length: int = os.environ.get("WINDOW_LENGTH", 15)
    polyorder: int = os.environ.get("POLYORDER", 6)

    def __post_init__(self):
        pass

    @classmethod
    def validate_params(cls, signal_length: int) -> tuple[int, int]:
        """Validate and adjust parameters based on signal length"""
        # Make window length odd
        window_length = cls.window_length
        if window_length % 2 == 0:
            window_length += 1

        # Ensure window length is smaller than signal length
        window_length = min(window_length, signal_length - 1)

        # Ensure polyorder is valid for window length
        polyorder = min(cls.polyorder, window_length - 1)
        return window_length, polyorder

    @classmethod
    def apply(cls, signal: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay filter to signal"""
        signal = np.array(signal)

        # Handle very short signals
        if len(signal) <= cls.polyorder + 2:
            return signal.copy()

        # Get validated parameters
        window_length, polyorder = cls.validate_params(len(signal))

        # Apply filter if parameters are valid
        if window_length > polyorder:
            return savgol_filter(signal, window_length, polyorder)

        # Return original if invalid
        return signal.copy()


@dataclasses.dataclass
class PeakDetector:
    height: float = os.environ.get("HEIGHT", 0.7)
    threshold: float | None = os.environ.get("THRESHOLD", None)
    distance: int| None = os.environ.get("DISTANCE", None)
    prominence: float = os.environ.get("PROMINENCE", 0.4)
    width: int | None = os.environ.get("WIDTH", None)
    wlen: int | None = os.environ.get("WLEN", None)
    rel_height: float = os.environ.get("REL_HEIGHT", 0.5)
    plateau_size: int | None = os.environ.get("PLATEAU_SIZE", None)

    def __post_init__(self) -> None:
        for k, v in self.__dict__.items():
            new_val = os.getenv(k.upper(), v)

            if not (new_val is None) and new_val.upper() == "NONE":
                new_val = None
            setattr(self, k, new_val)

    @classmethod
    def detect(cls, signal: np.ndarray) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Detect peaks in the signal using configured parameters"""
        return find_peaks(signal, **cls.__dict__)
