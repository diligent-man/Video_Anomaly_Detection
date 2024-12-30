import torch

from typing import Tuple


__all__ = [
    "load_video_v1",
    "load_video_v2",
    "load_video_v3"
]


def load_video_v1(path: str, output_format: str = "TCHW") -> torch.Tensor:
    import torchvision
    """
    :param path: path to video
    :param output_format: returned shape. Currently, THWC or TCHW
    :return: decoded video frames tensor
    Decode video with pyav, Pythonic binding for ffmpeg, as a backend
    """
    # Ignore audio frame and info in returned result
    frames, _, _ = torchvision.io.read_video(path, pts_unit="sec", output_format=output_format)
    return frames


def load_video_v2(path: str,
                  threads: int = 32,
                  thread_type: str = "slice",
                  device: str = "cuda",
                  output_shape: Tuple[int, int] = (224, 224),
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

        rgb: torch.Tensor = torch.stack([r, g, b], -1)
        rgb = rgb.clamp(0, 255).to(torch.uint8)
        return rgb

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
    stream_reader = torchaudio.io.StreamReader(path)
    decoder = stream_reader.get_src_stream_info(0).codec

    if device == "cpu":
        del __DEFAULT_DECODER_CONFIG["gpu"]
        del __DEFAULT_DECODER_CONFIG["resize"]
    else:
        decoder = __DECODERS[decoder]

    stream_reader.add_video_stream(
        frames_per_chunk=-1,
        buffer_chunk_size=-1,
        decoder=decoder,
        decoder_option=__DEFAULT_DECODER_CONFIG,
        hw_accel=device if device == "cuda" else None,
    )

    stream_reader.process_all_packets()

    # frames is in YUV444P format
    frames: torch.Tensor = stream_reader.pop_chunks()[0]
    frames = frames.to("cpu")
    frames = _yuv_to_rgb(frames)
    return frames
