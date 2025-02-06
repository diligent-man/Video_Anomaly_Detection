# Dirty workaround for module import, which violates PEP8: E402
import os
import sys
sys.path.append(os.path.join(os.getcwd(), "../../"))

import pathlib
import argparse
import warnings

from AI.src.utils import ConfigReader
from AI.src.utils.DotDict import DotDict
from AI.src.data.dataset.VideoDataset import VideoDataset
# from AI.src.data.dataset.VideoFolderDataset import VideoFolderDataset
warnings.filterwarnings("once")


def main(args: argparse.Namespace) -> None:
    # init dist environment
    # if config["Global"]["distributed"]:
    #     dist.init_parallel_env()
    #
    config: DotDict = ConfigReader(args.config).config

    import torch
    import time
    from AI.src.utils import load_video_v2, load_video_v1
    from AI.src.utils.saving import save_video
    ds: VideoDataset = VideoDataset(
        "/home/trong/Downloads/Dataset/VAD/UCF-Crime/Anomaly_videos/Abuse",
        "mp4",
        device="cpu",
        loader=load_video_v2
    )

    start = time.time()
    dl = torch.utils.data.DataLoader(ds, batch_size=16, num_workers=8, multiprocessing_context="fork", shuffle=False)
    path, data = next(iter(dl))
    print(data.shape)

    print(time.time() - start)
    # v1: 72.28s (1)
    # v2:
    return None


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config",
                      default=pathlib.Path("../config/teacher.json"),
                      type=str,
                      help="Path to config file"
                      )

    args = args.parse_args()
    main(args)
