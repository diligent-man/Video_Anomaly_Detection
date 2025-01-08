import torch
import torchaudio


__all__ = ["inspect_ffmpeg"]


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
    print("Avaialbe GPU:")
    print(torch.cuda.get_device_properties(0))
    return None
