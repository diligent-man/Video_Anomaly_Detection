import time
import multiprocessing as mp

from multiprocessing import Pool
from typing import List, Dict, Any, Tuple, Generator


import torch
import ffmpeg

from torch import Tensor
from torch.nn import Module
from torchvision.transforms import v2
from torchvision.io import decode_image, ImageReadMode
from torch.autograd.grad_mode import inference_mode


from ..utils import find_video_stream


__all__ = [
    "load_img", "extract_frames",
    "infer_for_test_v1", "infer_for_test_v2",
    "find_first_half_idx",
    "dispatch_infer"
]


global starter


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


def dispatch_infer(cache: Dict[str, Any],
                   model: Module,
                   device: str,
                   T_max: int,
                   overlap_ratio: float,
                   tolist: bool = True,
                   amp_cfg: Dict[str, Any] = None,
                   grad_ctx: inference_mode = inference_mode
                   ) -> Tuple[Tensor | List[float], Tensor | List[int]]:
    with Pool(min(32, cache["batch_worker"]), _init_proc, [mp.Value("d"), cache["batch_worker"]]) as pool:
        result: Tuple[List[float], List[int]] = pool.starmap(
            infer_for_test_v2,
            zip(cache["inp"],
                cache["label"],
                [model] * len(cache["inp"]),
                [device] * len(cache["inp"]),
                [T_max] * len(cache["inp"]),
                [overlap_ratio] * len(cache["inp"]),
                [tolist] * len(cache["inp"]),
                [amp_cfg] * len(cache["inp"]),
                [grad_ctx] * len(cache["inp"])
                )
        )
        return result


def find_first_half_idx(cur_frame_idx: int,
                        cum_frames: int,
                        total_frames: int,
                        T_max: int
                        ) -> Tuple[int, int]:
    if cur_frame_idx == total_frames - 1:
        # last iter
        start_idx: int = cur_frame_idx - cum_frames
        end_idx: int = start_idx + T_max // 2
    else:
        # others
        start_idx: int = cur_frame_idx - T_max
        end_idx: int = cur_frame_idx - (T_max // 2)
    return start_idx, end_idx


def infer_for_test_v1(inp: str | Tensor,
                      label: Tensor,
                      model: Module,
                      device: str = "cpu",
                      T_max: int = 30,
                      overlap_ratio: float = 0.5,
                      tolist: bool = True,
                      amp_cfg: Dict[str, Any] = None,
                      grad_ctx: inference_mode = inference_mode
                      ) -> Tuple[Tensor | List[float], Tensor | List[int]]:
    if amp_cfg is None:
        amp_cfg: Dict[str, Any] = {"enabled": True, "dtype": torch.float16}

    amp_cfg = {"device_type": device, **amp_cfg}

    with grad_ctx, torch.amp.autocast(**amp_cfg):
        if isinstance(inp, str):
            assert inp.endswith(".pt"), "Currently support .pt input file"
            inp = torch.load(inp, map_location="cpu", weights_only=False)  # (T,H,W,C)
            inp = v2.ToDtype(torch.float16, True)(inp)
            inp = inp.permute(0, -1, 1, 2).unsqueeze(0)

        # (B,T,C,H,W) -> (T,C,H,W)
        inp: Tensor = inp.squeeze()
        inp = inp.permute(1, 0, 2, 3).unsqueeze(0).unsqueeze(0)  # (T,C,H,W) -> (1,1,C,T,H,W)

        label: Tensor = label.squeeze(0)  # (T,)

        total_frames: int = label.shape[0]
        preds: Tensor = torch.zeros_like(label, dtype=torch.float16)

        cum_frames: int = 0
        step_preds: None | Tensor = None
        for i in range(total_frames):
            if cum_frames < T_max and i < total_frames - 1:
                # increment cum_frames until penultimate iter
                cum_frames += 1
            else:
                prev_cum_frames = cum_frames

                if cum_frames == T_max:
                    cum_frames = int(T_max * overlap_ratio) + 1

                if i == total_frames - 1:
                    # last step
                    step_preds: Tensor = model(inp[:, :, :, i - prev_cum_frames + 1:, ...].to(device)).preds  # (B, T)
                else:
                    # others
                    step_preds: Tensor = model(inp[:, :, :, i - prev_cum_frames:i, ...].to(device)).preds  # (B, T)

            if step_preds is not None:
                step_preds = step_preds.squeeze(0).to("cpu")

                # First half
                if preds[i - T_max: i - (T_max // 2)].equal(torch.zeros_like(preds[i - T_max: i - (T_max // 2)], dtype=preds.dtype)):
                    # first iter
                    first_half_start, first_half_end = i - T_max, i - (T_max // 2)
                    preds[first_half_start: first_half_end] += step_preds
                else:
                    # others
                    first_half_start, first_half_end = find_first_half_idx(i, prev_cum_frames, total_frames, T_max)
                    preds[first_half_start: first_half_end] = (preds[first_half_start: first_half_end] + step_preds) / 2

                # Second half
                second_half_end_idx: int = None if i == total_frames - 1 else i
                preds[first_half_end: second_half_end_idx] += step_preds

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
                      tolist: bool = True,
                      amp_cfg: Dict[str, Any] = None,
                      grad_ctx: inference_mode = inference_mode,
                      ) -> Tuple[Tensor | List[float], Tensor | List[int]]:
    if amp_cfg is None:
        amp_cfg: Dict[str, Any] = {"enabled": True, "dtype": torch.float16}

    amp_cfg = {"device_type": device, **amp_cfg}
    with grad_ctx, torch.amp.autocast(**amp_cfg):
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
                first_half_start, first_half_end = cur_frame_idx - T_max, cur_frame_idx - (T_max // 2)
                preds[first_half_start: first_half_end] += step_preds
            else:
                # others
                first_half_start, first_half_end = find_first_half_idx(cur_frame_idx, cum_frames, total_frames, T_max)
                preds[first_half_start: first_half_end] = (preds[first_half_start: first_half_end] + step_preds) / 2

            # Second half
            second_half_end_idx: int = None if cur_frame_idx == total_frames - 1 else cur_frame_idx
            preds[first_half_end: second_half_end_idx] += step_preds
    if tolist:
        preds: List[float] = preds.tolist()  # (B,T) -> (T,)
        label: List[int] = label.tolist()  # (B,T) -> (T,)
    return preds, label
########################################################################################################################


def _init_proc(shared_val: mp.Value, batch_worker: int) -> None:
    # ref: https://stackoverflow.com/a/70449572
    global starter
    starter = shared_val
    with starter.get_lock():
        if batch_worker <= 14:
            time.sleep(8)
        else:
            time.sleep(0)


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
        if cum_frames < T_max and cur_frame_idx < total_frames - 1:
            # increment cum_frames until penultimate iter
            cum_frames += 1
        else:
            prev_cum_frames: int = cum_frames

            # reset
            if cum_frames == T_max:
                cum_frames = int(T_max * overlap_ratio) + 1

            if cur_frame_idx == total_frames - 1:
                # last iter
                yield inp[:, :, :, cur_frame_idx - prev_cum_frames + 1:, ...], cur_frame_idx, prev_cum_frames
            else:
                # others
                yield inp[:, :, :, cur_frame_idx - prev_cum_frames: cur_frame_idx, ...], cur_frame_idx, prev_cum_frames
