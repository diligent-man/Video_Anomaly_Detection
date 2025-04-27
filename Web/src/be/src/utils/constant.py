import dataclasses
import os
import numpy as np
from typing import Dict, Any, Tuple
from scipy.signal import find_peaks, peak_widths, savgol_filter


@dataclasses.dataclass
class smooth_signal:
    window_length: int = int(os.environ.get("WINDOW_LENGTH", 15))
    polyorder: int = int(os.environ.get("POLYORDER", 6))

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
    prominence: float = os.environ.get("PROMINENCE", 0.4)
    width: int = None if os.environ.get("WIDTH") is None else int(os.environ.get("PEAK_WIDTH"))
    distance: int = None if os.environ.get("DISTANCE") is None else int(os.environ.get("DISTANCE"))
    threshold: float = None if os.environ.get("THRESHOLD") is None else int(os.environ.get("THRESHOLD"))
    wlen: int = None if os.environ.get("WLEN") is None else int(os.environ.get("WLEN"))
    rel_height: float = os.environ.get("REL_HEIGHT", 0.2)

    @classmethod
    def get_params(cls) -> Dict[str, Any]:
        """Get parameters dictionary for scipy.signal.find_peaks"""
        params = {
            "height": cls.height,
            "prominence": cls.prominence,
            "width": cls.width
        }

        # Only include optional parameters if they are not None
        if cls.distance is not None:
            params["distance"] = cls.distance
        if cls.threshold is not None:
            params["threshold"] = cls.threshold
        if cls.wlen is not None:
            params["wlen"] = cls.wlen
        return params

    @classmethod
    def detect(cls, signal: np.ndarray) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Detect peaks in the signal using configured parameters"""
        return find_peaks(signal, **cls.get_params())

    @classmethod
    def get_peak_regions(cls,
                         signal: np.ndarray,
                         peaks: np.ndarray
                         ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Calculate peak width regions using peak_widths"""
        widths, width_heights, left_ips, right_ips = peak_widths(signal, peaks, rel_height=cls.rel_height)
        return widths, width_heights, left_ips, right_ips
