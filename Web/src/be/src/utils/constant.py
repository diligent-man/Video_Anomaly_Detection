import dataclasses
import os
import numpy as np
from typing import Dict, Any
from scipy.signal import find_peaks, peak_widths, savgol_filter


@dataclasses.dataclass
class smooth_signal:
    window_length: int = int(os.environ.get("WINDOW_LENGTH", 11))
    polyorder: int = int(os.environ.get("POLYORDER", 3))

    def validate_params(self, signal_length: int) -> tuple[int, int]:
        """Validate and adjust parameters based on signal length"""
        # Make window length odd
        window_length = self.window_length
        if window_length % 2 == 0:
            window_length += 1

        # Ensure window length is smaller than signal length
        window_length = min(window_length, signal_length - 1)

        # Ensure polyorder is valid for window length
        polyorder = min(self.polyorder, window_length - 1)

        return window_length, polyorder

    def apply(self, signal: np.ndarray) -> np.ndarray:
        """Apply Savitzky-Golay filter to signal"""
        signal = np.array(signal)

        # Handle very short signals
        if len(signal) <= self.polyorder + 2:
            return signal.copy()

        # Get validated parameters
        window_length, polyorder = self.validate_params(len(signal))

        # Apply filter if parameters are valid
        if window_length > polyorder:
            return savgol_filter(signal, window_length, polyorder)

        # Return original if invalid
        return signal.copy()


@dataclasses.dataclass
class PeakDetector:
    height: float = float(os.environ.get("HIGH_THRESHOLD", 0.5))
    prominence: float = float(os.environ.get("LOW_THRESHOLD", 0.4))
    width: int = int(os.environ.get("PEAK_WIDTH", 1))
    distance: int = int(os.environ.get("DISTANCE", 5))
    threshold: float = float(os.environ.get("THRESHOLD", 0.5))
    wlen: int = int(os.environ.get("WLEN", 256))
    rel_height: float = float(os.environ.get("REL_HEIGHT", 0.5))

    def get_params(self) -> Dict[str, Any]:
        """Get parameters dictionary for scipy.signal.find_peaks"""
        params = {
            "height": self.height,
            "prominence": self.prominence,
            "width": self.width
        }

        # Only include optional parameters if they are not None
        if self.distance is not None:
            params["distance"] = self.distance
        if self.threshold is not None:
            params["threshold"] = self.threshold
        if self.wlen is not None:
            params["wlen"] = self.wlen
        return params

    def detect(self, signal: np.ndarray) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Detect peaks in the signal using configured parameters"""
        return find_peaks(signal, **self.get_params())

    def get_peak_regions(self,
                         signal: np.ndarray,
                         peaks: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate peak width regions using peak_widths"""
        return peak_widths(signal, peaks, rel_height=self.rel_height)
