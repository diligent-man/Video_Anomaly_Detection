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
        "h264", "mpeg4"
    ]

    __filters: Dict[str, Dict[str, Any]] = {
        "fps": {"fps": 25, "round": "up"},
        "scale": {"w": 256, "h": 256, "sws_flags": "lanczos"},
        "crop": {"out_w": 224, "out_h": 224, "exact": 1, "keep_aspect": 1},
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
        os.makedirs((pathlib.Path(self.__spath)).parent, exist_ok=True)

    @property
    def fpath(self) -> str:
        return self.__fpath

    @property
    def spath(self) -> str:
        return self.__spath

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

    def _preprocess(self) -> None:
        """
        Preprocess includes:
            a/ Resampling video with specified fps
            b/ Rescale frame
            c/ Central crop frame
            d/ Save video stream as output
        """
        probe_info: Dict[str, Any] = ffmpeg.probe(self.fpath)

        if self.__device == "cuda":
            stream = ffmpeg.input(self.__fpath, hwaccel="cuda")[(self._find_video_stream(probe_info["streams"]))]
        else:
            stream = ffmpeg.input(self.__fpath)[(self._find_video_stream(probe_info["streams"]))]

        for filter_name, kwargs in self.__filters.items():
            stream = stream.filter(filter_name, **kwargs)

        stream = stream.output(self.__spath, pix_fmt="rgb24", loglevel="quiet")
        stream = stream.overwrite_output()
        stream.run()

        if self.__device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()

    def _sampling_and_convert_tensor(self):
        """
        Split video into segments and perform sampling by interpolation
        """
        try:
            video: torch.Tensor = v2(self.__spath, device=self.__device)  # [T,H,W,C]
        except torch.cuda.OutOfMemoryError:
            video: torch.Tensor = v2(self.__spath, device="cpu")  # [T,H,W,C]

        total_frames: int = video.shape[0]
        seg_start_idx: torch.Tensor = torch.linspace(0, total_frames, self.__num_segments).clamp(0, total_frames).int()

        segments: None | torch.Tensor = None
        for i in range(0, len(seg_start_idx)-1):
            start, end = seg_start_idx[i].item(), seg_start_idx[i + 1].item()
            indices: torch.Tensor = torch.arange(start, end, device=video.device)

            frames: torch.Tensor = torch.index_select(video, 0, indices)
            frames = frames.permute(-1, 0, 1, 2).unsqueeze(0)

            inter_mode = "nearest-exact" if indices.shape[0] > self.__num_frames else "trilinear"
            frames = torch.nn.functional.interpolate(
                frames.type(torch.float32) if inter_mode == "trilinear" else frames,
                self.__num_frames,
                mode=inter_mode
                )

            segments = frames if segments is None else torch.vstack((segments, frames))

            if self.__device == "cuda":
                gc.collect()
                torch.cuda.empty_cache()

        extension: str = pathlib.Path(self.spath).name.split(".")[-1]
        torch.save(segments, self.spath.replace(extension, "pt"))
        os.remove(self.__spath)
        return None

    def __call__(self) -> None:
        self._preprocess()
        self._sampling_and_convert_tensor()
