import os
import sys
import pathlib

from typing import List
from multiprocessing import Pool
from argparse import ArgumentParser, Namespace

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), ".."))

from tqdm import tqdm

from AI.src.preprocessing import VideoPreprocessor
from AI.src.data.dataset import VideoFolderDataset
from AI.src.data.dataloader import DefaultDataLoader


def custom_collate_fn(batch) -> List[str]:
    paths: List[str] = list(map(lambda item: item[0], batch))
    return paths


def main(args: Namespace) -> None:
    ds_name: str = pathlib.Path(args.root).name
    pool: Pool = Pool(processes=args.processes)
    ds = VideoFolderDataset(args.root, "v6")
    dl = DefaultDataLoader(ds, args.batch_size, None, collate_fn=custom_collate_fn, drop_last=False)

    for fpaths in tqdm(dl, total=int(len(ds) / args.processes), desc=f"Dataset: {ds_name}", colour="red"):
        if args.device == "both":
            devices: List[str] = list(map(lambda i: "cpu" if i < len(fpaths) // 2 else "cuda", range(len(fpaths))))
        else:
            devices: List[str] = [args.device] * len(fpaths)

        preprocessors: List[VideoPreprocessor] = list(map(
            lambda fpath, device: VideoPreprocessor(fpath, args.save_root, ds_name, device), *(fpaths, devices))
        )

        pool.map(VideoPreprocessor.__call__, preprocessors)
    return None


if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--device",
                                 default="cpu",
                                 type=str,
                                 help="Device for preprocessing video with ffmpeg"
                                 )
    argument_parser.add_argument("--root",
                                 type=str,
                                 help="Root of dataset, which is read by VideoFolderDataset class"
                                 )
    argument_parser.add_argument("--save_root",
                                 default="out",
                                 type=str,
                                 help="Output root of preprocessed videos"
                                 )
    argument_parser.add_argument("--loader",
                                 default="v6",
                                 type=str,
                                 help="Video loader api"
                                 )
    argument_parser.add_argument("--batch_size",
                                 default=48,
                                 type=int,
                                 help="Batch size for dataloader"
                                 )
    argument_parser.add_argument("--processes",
                                 default=os.cpu_count(),
                                 type=int,
                                 help="Num processes for multiprocessing")

    parsed_args = argument_parser.parse_args()
    main(parsed_args)
