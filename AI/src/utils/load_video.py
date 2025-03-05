import os
import pathlib
from typing import Tuple, Dict, Callable

import torch
import torchaudio


def v1(path: str, output_format: str = "TCHW") -> torch.Tensor:
    """
    :param path: path to video
    :param output_format: returned shape. Currently, THWC or TCHW
    :return: decoded video frames tensor
    Decode video with pyav, Pythonic binding for ffmpeg, as a backend
    """
    import torchvision

    # Ignore audio frame and info in returned result
    frames, _, _ = torchvision.io.read_video(path, pts_unit="sec", output_format=output_format)
    return frames


def v2(path: str,
       fps: int = None,
       chunk_multiplier: int = 5,
       buffer_multiplier: int = 3,
       threads: int = 32,
       thread_type: str = "slice",
       device: str = "cuda",
       output_shape: Tuple[int, int] = (224, 224)
       ) -> torch.Tensor:
    import torchaudio
    """
    :param path: path to video
    :param threads: how many threads to be used in decoding
    :param thread_type: how to parallel decoding processing in backend. "frame" or "slice"
    :param device: device is used to decode video
    :param output_shape: returned shape in the format of (H, W)
    :param output_format: returned shape of tensor
    :return: decoded video frames tensor in shape (THWC)
    Decode video with ffmpeg as a backend
    """
    def _yuv_to_rgb(imgs: torch.Tensor) -> torch.Tensor:
        """
        Currently, HW decoder does not support colorspace conversion. Decoded frames are YUV format.
        The following function performs YUV to RGB conversion (and axis shuffling for plotting).

        # Warning from torchaudio
        # "The output format YUV420P is selected. This will be implicitly converted to YUV444P"
        # "Warning: The output format NV12 is selected. This will be implicitly converted to YUV444P"

        Ref: https://pytorch.org/audio/main/tutorials/nvdec_tutorial.html
        """
        imgs = imgs.to(torch.float)

        y: torch.Tensor = imgs[..., 0, :, :]
        u: torch.Tensor = imgs[..., 1, :, :]
        v: torch.Tensor = imgs[..., 2, :, :]

        r: torch.Tensor = 1.164 * (y - 16) + 1.596 * (v - 128)
        g: torch.Tensor = 1.164 * (y - 16) - 0.813 * (v - 128) - 0.392 * (u - 128)
        b: torch.Tensor = 1.164 * (y - 16) + 2.017 * (u - 128)

        imgs: torch.Tensor = torch.stack([r, g, b], -1)
        imgs = imgs.clamp(0, 255).to(torch.uint8)
        return imgs

    # Ignore audio frame and info in returned result
    __DECODERS = ["h264", "mpeg4"]
    __DECODERS = {codec: f"{codec}_cuvid" for codec in __DECODERS}

    # Check option by cmd: ffmpeg -h decoder=h264_cuvid
    __DEFAULT_DECODER_CONFIG = {
        "gpu": "0",
        "resize": "{}x{}".format(*tuple(reversed(output_shape))),  # w x h
        "threads": str(threads),
        "thread_type": thread_type
    }

    # Format should be left blank for automatic definition
    stream_reader: torchaudio.io.StreamReader = torchaudio.io.StreamReader(path)

    decoder: str = stream_reader.get_src_stream_info(0).codec
    fps: int = int(stream_reader.get_src_stream_info(0).frame_rate) if fps is None else fps

    if device == "cpu":
        del __DEFAULT_DECODER_CONFIG["gpu"]
        del __DEFAULT_DECODER_CONFIG["resize"]
    else:
        decoder = __DECODERS[decoder]

    stream_reader.add_video_stream(
        fps * chunk_multiplier,
        fps * buffer_multiplier,
        decoder=decoder,
        decoder_option=__DEFAULT_DECODER_CONFIG,
        hw_accel=device if device == "cuda" else None
    )
    video: torch.Tensor | None = None
    for chunk in stream_reader.stream():
        frames = chunk[0]

        # read frames is in YUV444P format
        try:
            frames = _yuv_to_rgb(frames)
        except torch.cuda.OutOfMemoryError:
            frames = _yuv_to_rgb(frames.to("cpu"))

        frames = frames.to("cpu")
        video = frames if video is None else torch.vstack([video, frames])
    return video


def v3(path: str):
    """
    :param path: path to video
    :return: InputStream object
    """
    import torchaudio
    stream_reader: torchaudio.io.StreamReader = torchaudio.io.StreamReader(path)
    stream_info = stream_reader.get_src_stream_info(0)
    return stream_info


def v4(path: str) -> torch.Tensor:
    """
    Load video in .pt format
    """
    path = pathlib.Path(path)
    assert os.path.isfile(path), ValueError(f"Invalid path, Get {path}")
    assert path.suffix == ".pt", ValueError(f"Invalid file extension, Get {path}")
    return torch.load(path, weights_only=True)


def v5(path: str) -> torch.Tensor:
    """
    Create pseudo-tensor for dev stage
    """
    stream_reader: torchaudio.io.StreamReader = torchaudio.io.StreamReader(path)
    print(stream_reader)

    T = torch.randint(low=10, high=50, size=(1,)).item()
    pseudo_tensor: torch.Tensor = torch.rand(1, 3, T, 224, 224)
    return pseudo_tensor


def v6(path: str) -> str:
    """
    :param path: path to video
    :return path to video
    """
    return path


video_loader: Dict[str, Callable] = {
    "v1": v1,
    "v2": v2,
    "v3": v3,
    "v4": v4,
    "v5": v5,
    "v6": v6
}

__all__ = [video_loader]
