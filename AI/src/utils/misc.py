import os
import re
import sys
import warnings
import platform
from importlib import metadata
from typing import Tuple, Dict, Any, List

import torch
import torchaudio
from matplotlib import pyplot as plt

from . import DotDict, ANSIColor



__all__ = [
    "get_amp_cfg",
    "get_services",
    "visualize_lr",
    "inspect_ffmpeg",
    "check_version",
    "make_border",
    "multiple_replace"
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
            os.add_dll_directory(path)


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


########################################################################################################################
TORCH_2_4 = check_version(torch.__version__, "2.4.0")
MACOS, LINUX, WINDOWS = (platform.system() == x for x in ["Darwin", "Linux", "Windows"])  # environment booleans
