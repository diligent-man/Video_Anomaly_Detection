from typing import List, Dict, Any, Tuple, Generator

import torch
import ffmpeg

from torch import Tensor
from torch.nn import Module
from torchvision.transforms import v2
from torchvision.io import decode_image, ImageReadMode

from ..utils import find_video_stream

__all__ = ["load_img", "extract_frames", "infer_for_test_v1", "infer_for_test_v2"]


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


def infer_for_test_v1(inp: str | Tensor,
                      label: Tensor,
                      model: Module,
                      device: str = "cpu",
                      T_max: int = 30,
                      overlap_ratio: float = 0.5,
                      tolist: bool = True
                      ) -> Tuple[Tensor | List[float], Tensor | List[int]]:
    with torch.inference_mode(), torch.amp.autocast(device_type=device, enabled=True, dtype=torch.float16):
        if isinstance(inp, str):
            assert inp.endswith(".pt"), "Currently support .pt input file"
            inp = torch.load(inp, map_location="cpu", weights_only=False)  # (T,H,W,C)
            inp = v2.ToDtype(torch.float16, True)(inp)
            inp = inp.permute(0, -1, 1, 2).unsqueeze(0)

        # (B,T,C,H,W) -> (T,C,H,W)
        inp: Tensor = inp.squeeze()
        inp = inp.permute(1, 0, 2, 3).unsqueeze(0).unsqueeze(0)  # (T,C,H,W) -> (1,1,C,T,H,W)

        label: Tensor = label.squeeze(0)  # (T,)

        cum_frames: int = 0
        step_preds: None | Tensor = None
        total_frames: int = label.shape[0]
        preds: Tensor = torch.zeros_like(label, dtype=torch.float16)
        for i in range(total_frames):
            if i < T_max or cum_frames < T_max:
                cum_frames += 1
            else:
                # step when accumulate sufficient frames
                step_preds: Tensor = model(inp[:, :, :, i - cum_frames:i, ...].to(device)).preds  # (B, S)
                cum_frames = int(T_max * overlap_ratio) + 1

            # Last step
            if i == total_frames - 1:
                step_preds: Tensor = model(inp[:, :, :, i - cum_frames + 1:i, ...].to(device)).preds  # (B, S)

            if step_preds is not None:
                step_preds = step_preds.squeeze(0).to("cpu")

                # First half
                if preds[i - T_max: i - (T_max // 2)].equal(
                        torch.zeros_like(preds[i - T_max: i - (T_max // 2)], dtype=preds.dtype)
                ):
                    # first iter
                    preds[i - T_max: i - (T_max // 2)] += step_preds
                else:
                    if i == total_frames - 1:
                        # last iter
                        preds[i - cum_frames + 1:
                              i - cum_frames + 1 + (T_max // 2)
                              ] = (preds[i - cum_frames + 1:
                                         i - cum_frames + 1 + (T_max // 2)
                                         ] + step_preds) / 2
                    else:
                        # others
                        preds[i - T_max:
                              i - (T_max // 2)
                              ] = (preds[i - T_max:
                                         i - (T_max // 2)
                                         ] + step_preds) / 2

                # Second half
                if i == total_frames - 1:
                    # last iter
                    preds[i - cum_frames + 1 + (T_max // 2):] += step_preds
                else:
                    preds[i - (T_max // 2): i] += step_preds

                # Reset
                step_preds = None
    if tolist:
        preds: List[float] = preds.tolist()
        label: List[int] = label.tolist()
    return preds, label


def infer_for_test_v2(inp: str | Tensor,
                      label: Tensor,
                      model: Module,
                      device: str = "cpu",
                      T_max: int = 30,
                      overlap_ratio: float = 0.5,
                      tolist: bool = True
                      ) -> Tuple[Tensor | List[float], Tensor | List[int]]:
    with torch.inference_mode(), torch.amp.autocast(device_type=device, enabled=True, dtype=torch.float16):
        label: Tensor = label.squeeze(0)  # (B,T) -> (T,)

        total_frames: int = label.shape[0]
        preds: Tensor = torch.zeros_like(label, dtype=torch.float16)

        for inp, cur_frame_idx, cum_frames in _get_segment(inp, label, T_max, overlap_ratio):
            step_preds = model(inp.to(device)).preds  # (B, T)
            step_preds = step_preds.squeeze(0).to("cpu")

            # First half
            if preds[cur_frame_idx - T_max: cur_frame_idx - (T_max // 2)].equal(
                    torch.zeros_like(preds[cur_frame_idx - T_max: cur_frame_idx - (T_max // 2)], dtype=preds.dtype)
            ):
                # first iter
                preds[cur_frame_idx - T_max: cur_frame_idx - (T_max // 2)] += step_preds
            else:
                if cur_frame_idx == total_frames - 1:
                    # last iter
                    preds[cur_frame_idx - cum_frames + 1:
                          cur_frame_idx - cum_frames + 1 + (T_max // 2)
                          ] = (preds[cur_frame_idx - cum_frames + 1:
                                     cur_frame_idx - cum_frames + 1 + (T_max // 2)
                                     ] + step_preds) / 2
                else:
                    # others
                    preds[cur_frame_idx - T_max:
                          cur_frame_idx - (T_max // 2)
                          ] = (preds[cur_frame_idx - T_max:
                                     cur_frame_idx - (T_max // 2)
                                     ] + step_preds) / 2

            # Second half
            if cur_frame_idx == total_frames - 1:
                # last iter
                preds[cur_frame_idx - cum_frames + 1 + (T_max // 2):] += step_preds
            else:
                preds[cur_frame_idx - (T_max // 2): cur_frame_idx] += step_preds
    if tolist:
        preds: List[float] = preds.tolist()  # (B,S) -> (S,)
        label: List[int] = label.tolist()  # (B,S) -> (S,)
    return preds, label


def _get_segment(inp: str | Tensor,
                 label: Tensor,
                 T_max: int,
                 overlap_ratio: float
                 ) -> Generator:
    if isinstance(inp, str):
        assert inp.endswith(".pt"), "Currently support .pt input file"
        inp = torch.load(inp, map_location="cpu", weights_only=False)  # (T,H,W,C)
        inp = v2.ToDtype(torch.float16, True)(inp)
        inp = inp.permute(0, -1, 1, 2).unsqueeze(0)

    # (B,T,C,H,W) -> (T,C,H,W)
    inp: Tensor = inp.squeeze()
    inp = inp.permute(1, 0, 2, 3).unsqueeze(0).unsqueeze(0)  # (T,C,H,W) -> (1,1,C,T,H,W)

    label: Tensor = label.squeeze()  # (T,)
    total_frames: int = label.shape[0]

    cum_frames: int = 0
    for cur_frame_idx in range(total_frames):
        if cur_frame_idx < T_max or cum_frames < T_max:
            cum_frames += 1
        else:
            # others
            prev_cum_frames = cum_frames  # cache
            cum_frames = int(T_max * overlap_ratio) + 1  # reset
            yield inp[:, :, :, cur_frame_idx - prev_cum_frames: cur_frame_idx, ...], cur_frame_idx, prev_cum_frames

        # last step
        if cur_frame_idx == total_frames - 1:
            yield inp[:, :, :, cur_frame_idx - cum_frames + 1:cur_frame_idx, ...], cur_frame_idx, cum_frames
