from typing import List, Dict, Any


import torch
import ffmpeg

from torch import Tensor
from torchvision.transforms import v2
from torchvision.io import decode_image, ImageReadMode


from ..utils import find_video_stream


__all__ = ["load_img", "extract_frames"]


def load_img(img_lst: List[str], dtype: torch.dtype, device: str) -> Tensor:
    imgs: None | Tensor = None
    for i in img_lst:
        if imgs is None:
            imgs = decode_image(i, ImageReadMode.RGB).unsqueeze(0)
        else:
            imgs = torch.cat((imgs, decode_image(i, ImageReadMode.RGB).unsqueeze(0)), 0)
    # (T,C,H,W) -> (C,T,H,W)
    imgs = imgs.permute(1, 0, 2, 3).unsqueeze(0).unsqueeze(0)
    imgs = v2.ToDtype(dtype, True)(imgs)
    return imgs.to(device)


def extract_frames(video_path: str, tmp_dir: str) -> None:
    print(f"Extracting frames from {video_path} to {tmp_dir}...")
    _filters: Dict[str, Dict[str, Any]] = {
        "scale": {"w": 320, "h": 320, "sws_flags": "neighbor"},
        "crop": {"out_w": 224, "out_h": 224, "exact": 1, "keep_aspect": 1},
    }

    probe_info: Dict[str, Any] = ffmpeg.probe(video_path)
    stream = find_video_stream(probe_info["streams"])

    stream = ffmpeg.input(video_path)[stream]

    for filter_name, kwargs in _filters.items():
        stream = stream.filter(filter_name, **kwargs)

    stream = stream.output(f"{tmp_dir}/frame_%08d.jpg", pix_fmt="rgb24", loglevel="error")
    stream = stream.overwrite_output()
    stream.run()
    return None
