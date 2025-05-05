import os
import re
import sys
import warnings
import platform
from importlib import metadata
from typing import Tuple, Dict, Any, List


import torch
import torchaudio
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from . import DotDict, ANSIColor


__all__ = [
    "get_amp_cfg",
    "get_services",
    "visualize_lr",
    "inspect_ffmpeg",
    "check_version",
    "make_border",
    "multiple_replace",
    "draw_anomaly_graph"
]


"""
Manually call the _init_dll_path method to ensure that the system path is searched for FFMPEG.
Calling torchaudio._extension.utils._init_dll_path does not work because
it is initializing the torchadio module prematurely or something.

See: https://github.com/pytorch/audio/issues/3789
"""
if sys.platform == "win32":
    print("Initializing DLL path for Windows")
    for path in os.environ.get("Path", "").split(";"):
        if os.path.exists(path):
            dll_path = os.path.abspath(path)
            os.add_dll_directory(dll_path)


def inspect_ffmpeg() -> None:
    print("FFmpeg Library versions:")
    for k, ver in torchaudio.utils.ffmpeg_utils.get_versions().items():
        print(f"{k}:\t{'.'.join(str(v) for v in ver)}")
    print()

    print("Available NVENC Encoders:")
    for k in torchaudio.utils.ffmpeg_utils.get_video_encoders().keys():
        if "nvenc" in k:
            print(f" - {k}")
    print()
    print("Available GPU:")
    print(torch.cuda.get_device_properties(0))
    return None


def visualize_lr(optimizer: torch.optim.Optimizer,
                 scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                 mode: str = "update_per_batch",
                 epochs: int = 10,
                 dataloader_len: int = 10
                 ) -> None:
    plt.switch_backend("tkagg")
    
    lr = []
    if mode == "update_per_epoch":
        for epoch in range(epochs):
            for i in range(dataloader_len):
                optimizer.zero_grad()
                optimizer.step()

            if scheduler is not None:
                scheduler.step()
                lr.append(scheduler.get_last_lr().pop())
            else:
                lr.append(optimizer.state_dict()["param_groups"][0]["lr"])
        plt.plot(range(epochs), lr)
        plt.xlabel("Epochs")
    elif mode == "update_per_batch":
        for epoch in range(epochs):
            for i in range(dataloader_len):
                optimizer.zero_grad()
                optimizer.step()

                if scheduler is not None:
                    scheduler.step()
                    lr.append(scheduler.get_last_lr().pop())
                else:
                    lr.append(optimizer.state_dict()["param_groups"][0]["lr"])
        plt.plot(range(epochs * dataloader_len), lr)
        plt.xlabel("Iterations (Epochs * Dataloader_len)")

    plt.ylabel("Learning rate")
    plt.title(f"Optim: {optimizer.__class__.__name__}, Scheduler: {scheduler.__class__.__name__}")
    plt.show()
    return None


def get_amp_cfg(config: DotDict) -> Tuple[Dict[str, Any], None | torch.GradScaler]:
    scaler: None | torch.GradScaler = None
    device: str = config.Global.get("device", "cpu")
    use_amp: bool = config.Global.get("use_amp", False)

    if use_amp:
        # if device == "cuda" else torch.bfloat16  # cpu also use torch.float16 ???
        amp_dtype: torch.dtype = torch.float16

        # Currently use default arg for GradScaler
        if device == "cuda":
            scaler: torch.GradScaler = torch.amp.GradScaler(device, enabled=use_amp)
    else:
        amp_dtype: torch.dtype = torch.float32

    autocast_config: Dict[str, Any] = {"device_type": device, "dtype": amp_dtype}
    return autocast_config, scaler


def get_services(config: DotDict) -> List[str]:
    services: List[str] = []
    service_config: List[Dict[str, Any]] = config.Services

    if service_config is None:
        print("No additional service is specified")
    else:
        for service in service_config:
            apply_status = service.get("apply", False)

            if apply_status:
                services.append(service.name)
            else:
                setattr(service, "apply", False)
    return services


def check_version(
    current: str = "0.0.0",
    required: str = "0.0.0",
    hard: bool = False
) -> bool:
    """
    Adopted from Ultralytics
    Check current version against the required version or range.

    Args:
        current (str): Current version or package name to get version from.
        required (str): Required version or range (in pip-style format).
        hard (bool, optional): If True, raise an AssertionError if the requirement is not met.

    Returns:
        (bool): True if requirement is met, False otherwise.

    Example:
        ```python
        # Check if current version is exactly 22.04
        check_version(current="22.04", required="==22.04")

        # Check if current version is greater than or equal to 22.04
        check_version(current="22.10", required="22.04")  # assumes '>=' inequality if none passed

        # Check if current version is less than or equal to 22.04
        check_version(current="22.04", required="<=22.04")

        # Check if current version is between 20.04 (inclusive) and 22.04 (exclusive)
        check_version(current="21.10", required=">20.04,<22.04")
        ```
    """
    if not current:  # if current is '' or None
        warnings.warn(f"WARNING ⚠️ invalid check_version({current}, {required}) requested, please check values.")
        return True
    elif not current[0].isdigit():  # current is package name rather than version string, i.e. current='ultralytics'
        try:
            current = metadata.version(current)  # get version string from package name
        except metadata.PackageNotFoundError as e:
            if hard:
                raise ModuleNotFoundError(f"WARNING ⚠️ {current} package is required but not installed") from e
            else:
                return False

    if not required:  # if required is '' or None
        return True

    if "sys_platform" in required and (  # i.e. required='<2.4.0,>=1.8.0; sys_platform == "win32"'
        (WINDOWS and "win32" not in required)
        or (LINUX and "linux" not in required)
        or (MACOS and "macos" not in required and "darwin" not in required)
    ):
        return True


def make_border(headline: str) -> Tuple[str, str]:
    MAX_LINE_LEN: int = 80

    if len(headline) > MAX_LINE_LEN:
        headline = headline
    else:
        headline = f"{'-' * (MAX_LINE_LEN - len(headline))}"\
                   f"  {headline}  "\
                   f"{'-' * (MAX_LINE_LEN - len(headline))}"

    top: str = f"{ANSIColor().CYAN}{headline}{ANSIColor().RESET}"
    bottom: str = f"{ANSIColor().CYAN}{'-' * len(headline)}{ANSIColor().RESET}\n\n"
    return top, bottom


def multiple_replace(string: str, ref_dict: Dict[str, str]) -> str:
    ref_keys: List[str] = sorted(ref_dict, key=len, reverse=True)
    pattern: re.Pattern = re.compile("|".join([re.escape(k) for k in ref_keys]), flags=re.DOTALL)
    string = pattern.sub(lambda x: ref_dict[x.group(0)], string)
    return string


def calculate_iou(ranges1, ranges2):
    def to_set(ranges):
        frames = set()
        for start, end in ranges:
            frames.update(range(start, end + 1))
        return frames

    set1 = to_set(ranges1)
    set2 = to_set(ranges2)

    intersection = len(set1 & set2)
    union = len(set1 | set2)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def Overlap_ratio(ranges1, ranges2):
    def to_set(ranges):
        frames = set()
        for start, end in ranges:
            frames.update(range(start, end + 1))
        return frames

    set1 = to_set(ranges1)
    set2 = to_set(ranges2)

    intersection = len(set1 & set2)
    union = len(set1)

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return intersection / union

def draw_anomaly_graph(preds=None, anomaly_ranges=None, video_name="Anomaly Graph", save_path=None,
                       smooth_pred=None, smooth_label="Smoothed Pred", smooth_color="blue",
                       additional_anomaly_ranges=None, anomaly_color="red", additional_anomaly_color="green",
                       peaks=None, iou=None, Overlap=None):
    """
    Draws an anomaly graph with optional smoothed predictions and additional anomaly ranges.

    Args:
        preds (list or np.ndarray, optional): The prediction scores. If None, no prediction line is drawn.
        anomaly_ranges (list of tuples, optional): List of (start, end) for anomaly regions. Defaults to None.
        video_name (str, optional): Title of the graph. Defaults to "Anomaly Graph".
        save_path (str, optional): Path to save the plot. Defaults to None.
        smooth_pred (list or np.ndarray, optional): Smoothed prediction scores. Defaults to None.
        smooth_label (str, optional): Label for the smoothed line. Defaults to "Smoothed Pred".
        smooth_color (str, optional): Color for the smoothed line. Defaults to "blue".
        additional_anomaly_ranges (list of tuples, optional): Additional anomaly regions. Defaults to None.
        anomaly_color (str, optional): Color for the primary anomaly regions. Defaults to "red".
        additional_anomaly_color (str, optional): Color for the additional anomaly regions. Defaults to "green".
        peaks (list, optional): Indices of detected peaks. Defaults to None.
        iou (bool or float, optional): If True, calculate IoU between anomaly_ranges and additional_anomaly_ranges.
                                      If float, use the provided IoU value. Defaults to None.
        Overlap (bool or float, optional): If True, calculate overlap ratio between anomaly_ranges and additional_anomaly_ranges.
    """
    plt.figure(figsize=(14, 5))
    legend_elements = []

    
    
    # Calculate IoU if requested
    if iou is True and anomaly_ranges is not None and additional_anomaly_ranges is not None:
        iou = calculate_iou(anomaly_ranges, additional_anomaly_ranges)

    if Overlap is True and anomaly_ranges is not None and additional_anomaly_ranges is not None:
        overlap_ratio = Overlap_ratio(anomaly_ranges, additional_anomaly_ranges)
    
    # Plot predictions if provided
    if preds is not None:
        plt.plot(preds, label="Pred", color='blue')

    # Plot smoothed predictions if provided,
    if smooth_pred is not None:
        plt.plot(smooth_pred, label=smooth_label, color=smooth_color, linewidth=1.0, zorder = 3)
    
    # Add primary anomaly regions if provided
    if anomaly_ranges is not None:
        for i, (start, end) in enumerate(anomaly_ranges):
            if i == 0:  # Add label only for the first region
                plt.axvspan(start, end, color=anomaly_color, alpha=0.3, label="Labeled Anomaly Region", zorder =1)
            else:
                plt.axvspan(start, end, color=anomaly_color, alpha=0.3, zorder = 1)

    # Add additional anomaly regions if provided
    if additional_anomaly_ranges is not None:
        for i, (start, end) in enumerate(additional_anomaly_ranges):
            if i == 0:  # Add label only for the first region
                plt.axvspan(start, end, color=additional_anomaly_color, alpha=0.3, label="Detected Anomaly Region", zorder = 2)
            else:
                plt.axvspan(start, end, color=additional_anomaly_color, alpha=0.3, zorder = 2)
    
    # Plot peaks if provided and smooth_pred is also provided
    if peaks is not None and len(peaks) > 0 and smooth_pred is not None:
        # Get the values at the peak positions
        peak_values = np.array(smooth_pred)[peaks]
        plt.scatter(peaks, peak_values, color='darkred', s=20, label="Detected Peaks", 
                    marker='x', alpha=1.0, linewidths=0.7, zorder = 4)
    
    # Create title with video name (without IoU)
    plt.title(video_name, fontsize=16)
    plt.xlabel("Frame", fontsize=12)
    plt.ylabel("Anomaly Score", fontsize=12)
    
    # Get handles and labels for the legend
    handles, labels_ = plt.gca().get_legend_handles_labels()
    
    # Add IoU to legend if provided
    if iou is not None and isinstance(iou, (int, float)):
        # Create a custom handle for IoU that doesn't appear in the plot
        iou_patch = Patch(color='none', label=f"IoU: {iou:.3f}")
        handles.append(iou_patch)
        labels_.append(f"IoU: {iou:.3f}")
    
    if overlap_ratio is not None and isinstance(overlap_ratio, (int, float)):
        # Create a custom handle for IoU that doesn't appear in the plot
        overlap_ratio_patch = Patch(color='none', label=f"Overlap ratio: {overlap_ratio:.3f}")
        handles.append(overlap_ratio_patch)
        labels_.append(f"Overlap ratio: {overlap_ratio:.3f}")
    
    # Create legend with unique items
    by_label = dict(zip(labels_, handles))
    
    # Place legend outside the plot
    if by_label:  # Only add legend if there are items to show
        plt.legend(by_label.values(), by_label.keys(), 
                  loc='upper left', bbox_to_anchor=(1.05, 1), 
                  borderaxespad=0., frameon=True)
    
    plt.grid(False)
    plt.xlim(left=0)
    plt.ylim(0, 1)
    plt.tick_params(axis='y', which='both', direction='in')

    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        plt.close()
    else:
        plt.show()
########################################################################################################################


TORCH_2_4 = check_version(torch.__version__, "2.4.0")
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans
