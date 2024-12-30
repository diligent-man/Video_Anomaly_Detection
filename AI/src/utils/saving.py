import os

from pathlib import Path
from typing import List


import torch
import torchvision

from PIL.Image import Image
from torchvision.transforms import v2


__all__ = ["save_batch_images", "save_video"]


def save_batch_images(batch: torch.Tensor,
                      save_dir: str | Path,
                      name: str = "",
                      indexing_img: bool = True,
                      extension: str = "jpg",
                      idx: List[int] = None
                      ) -> None:
    """
    :param batch: shape (T, C, H, W) with C = (1, 3)
    :param save_dir: save directory
    :param name: file name
    :param indexing_img: append auto index for saved image
    :param extension: image extension
    :param idx: user-defined indices for saved image
    :return: None
    """
    assert extension in ("jpg", "png"), ValueError("Image extension is currently not supported")

    if idx:
        assert len(idx) == len(batch), "Length of idx must be equivalent to batch size"

    if not os.path.exists(save_dir):
        print("Save path for saving batch images not exist. Creating ...")
        os.makedirs(save_dir, exist_ok=True)
    else:
        print("Save path for saving batch images existed so not create the new one.")

    for i, img in enumerate(batch):
        img: Image = v2.ToPILImage()(img)

        if idx:
            save_name: str = f"{name}_{idx[i]}"
        else:
            save_name: str = f"{name}_{i}" if indexing_img else f"{name}"

        save_name = f"{save_name}.{extension}"
        img.save(fp=os.path.join(save_dir, save_name))
    return None


def save_video(frames: torch.Tensor,
               save_dir: str,
               fps: int = 24,
               name: str = "output_video"
               ) -> None:
    """
    :param frames: shape (T, H, W, 3) and in range [0, 255] in lieu of [0, 1] due to write_video fn
    :param save_dir: save directory
    :param fps: frame per sec
    :param name: file name
    :return: None
    """
    if not os.path.exists(save_dir):
        print(f"{save_dir} not exist. Creating ...")
        os.makedirs(save_dir, exist_ok=True)
    else:
        print(f"{save_dir} existed so not create the new one.")

    frames = frames.to("cpu").type(torch.uint8)

    filename = name if name.endswith(".mp4") else f"{name}.mp4"
    filename = os.path.join(save_dir, filename)

    torchvision.io.write_video(filename, frames, fps)
    return None
