import warnings

import torch
import torchaudio

from . import DotDict
from typing import Tuple, Dict, Any, List
from matplotlib import pyplot as plt

plt.switch_backend("tkagg")


__all__ = [
    "get_amp_cfg",
    "get_services",
    "visualize_lr",
    "inspect_ffmpeg"
]


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
    use_amp: bool = config.Global.get("use_amp", False)
    use_grad_scaler: bool = config.Global.get("use_grad_scaler", False)

    scaler: None | torch.GradScaler = None
    device: str = config.Global.get("device", "cpu")

    if use_amp:
        amp_dtype: torch.dtype = config.Global.get("amp_dtype", torch.float16)  # cpu also use torch.float16 ???

        # currently use default arg for GradScaler
        if use_grad_scaler:
            scaler: torch.GradScaler = torch.amp.GradScaler(device, enabled=use_amp)
        else:
            warnings.warn("use_grad_scaler should be True if use_amp=True")
    else:
        amp_dtype: torch.dtype = torch.float32

    autocast_config: Dict[str, Any] = {"device": device, "dtype": amp_dtype}
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
