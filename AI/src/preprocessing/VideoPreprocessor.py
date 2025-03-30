import gc
import os
import pathlib
from typing import List, Any, Dict

import torch
import ffmpeg

from ..utils.load_video import v2


__all__ = ["VideoPreprocessor"]


class VideoPreprocessor(object):
    # Add more if necessary
    __CODECS = [
        "h264", "mpeg4", "vp9", "mjpeg", "av1"
    ]

    __filters: Dict[str, Dict[str, Any]] = {
        "fps": {"fps": 15, "round": "up"},
        "scale": {"w": 320, "h": 320, "sws_flags": "neighbor"},
        # "crop": {"out_w": 224, "out_h": 224, "exact": 1, "keep_aspect": 1},
    }

    def __init__(self,
                 fpath: str,
                 save_root: str,
                 dataset_name: str,
                 device: str,
                 num_segments: int = 32,
                 num_frames: int = 30,
                 filters: Dict[str, Dict[str, Any]] = None,
                 ) -> None:
        super(VideoPreprocessor, self).__init__()
        self.__fpath: str = fpath
        self.__spath: str = self._make_spath(dataset_name, save_root)
        self.__device: str = device
        self.__num_segments: int = num_segments
        self.__num_frames: int = num_frames
        self.__filters = filters or self.__filters
        self.__is_labeled: bool = self._is_labeled(dataset_name)
        os.makedirs((pathlib.Path(self.__spath)).parent, exist_ok=True)

    @property
    def fpath(self) -> str:
        return self.__fpath

    @property
    def spath(self) -> str:
        return self.__spath

    @property
    def is_label(self) -> bool:
        return self.__is_labeled

    def _is_labeled(self, ds_name: str) -> bool:
        flag: bool = False
        path_components: List[str] = self.__fpath.split(os.sep)

        if path_components[path_components.index(ds_name) + 1] == "labeled":
            flag = not flag
        return flag

    def _find_video_stream(self, streams: Dict[str, Any]) -> str:
        stream: Dict[str, Any]

        for i, stream in enumerate(streams):
            if stream["codec_name"] in self.__CODECS:
                return str(i)

    def _make_spath(self, dataset_name: str, save_root: str) -> str:
        ds_name_idx = self.__fpath.split(os.sep).index(dataset_name)

        path_components: List[str] = self.__fpath.split(os.sep)
        path_components.insert(ds_name_idx, save_root)

        spath: str = f"{os.sep}".join(path_components)
        return spath

    def stage_one(self, run_async: bool = False) -> None:
        """
        Stage one includes:
            a/ Resampling video with specified fps
            b/ Rescale frame
            c/ Central crop frame (temporarily disabled)
            d/ Save video stream as output
        """
        if not os.path.exists(self.__spath):
            if self.__is_labeled:
                self.__filters.pop("fps", None)

            probe_info: Dict[str, Any] = ffmpeg.probe(self.fpath)
            stream = self._find_video_stream(probe_info["streams"])

            try:
                stream = ffmpeg.input(self.__fpath, hwaccel="cuda")[stream] if self.__device == "cuda" else \
                    ffmpeg.input(self.__fpath)[stream]

                for filter_name, kwargs in self.__filters.items():
                    stream = stream.filter(filter_name, **kwargs)

                stream = stream.output(self.__spath, pix_fmt="rgb24", loglevel="verbose")
                stream = stream.overwrite_output()
                stream.run_async() if run_async else stream.run()
            except ffmpeg.Error as e:
                print(f"File: {self.__fpath} get {e}\n so ignore it")

                if os.path.exists(self.__spath):
                    os.remove(self.__spath)

    def stage_two(self,
                  del_prev_result: bool = False
                  ) -> None:
        """
        Stage two includes:
            a/ Split video into segments
            b/ Temporal sampling by interpolation (up/ down scaling)
            c/ Save video stream as .pt file
         and
        """
        ext: str = pathlib.Path(self.__spath).suffix
        if not os.path.isfile(self.__spath.replace(ext, ".pt")) and os.path.exists(self.__spath):
            video: torch.Tensor = v2(self.__spath, device=self.__device)  # [T,H,W,C] in cpu device

            total_frames: int = video.shape[0]
            seg_start_idx: torch.Tensor = torch.linspace(
                0, total_frames, self.__num_segments
            ).clamp(0, total_frames).int()

            save_tensor: None | torch.Tensor = None
            if not self.__is_labeled:
                for i in range(0, len(seg_start_idx)-1):
                    start, end = seg_start_idx[i].item(), seg_start_idx[i + 1].item()

                    indices: torch.Tensor = torch.arange(start, end, device=video.device, dtype=torch.int32)
                    inter_mode = "nearest-exact" if indices.shape[0] > self.__num_frames else "trilinear"

                    frames: torch.Tensor = torch.index_select(video, 0, indices)
                    frames = frames.to("cpu").permute(-1, 0, 1, 2).unsqueeze(0).to(self.__device)
                    frames = torch.nn.functional.interpolate(
                        frames.type(torch.float32) if inter_mode == "trilinear" else frames,
                        (self.__num_frames, *frames.shape[-2:]),
                        mode=inter_mode
                        ).to("cpu").type(torch.uint8)

                    save_tensor = frames if save_tensor is None else torch.vstack((save_tensor, frames))
            else:
                save_tensor = video

            torch.save(save_tensor, self.__spath.replace(ext, ".pt"))
            torch.serialization.add_safe_globals([save_tensor])

            if del_prev_result:
                os.remove(self.__spath)

            del video
            gc.collect()
            torch.cuda.empty_cache()
            torch.serialization.clear_safe_globals()
        return None
