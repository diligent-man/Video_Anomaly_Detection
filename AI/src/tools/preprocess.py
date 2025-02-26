"""Script for preprocessing train video."""
import os
import sys
import pathlib
import time

from functools import partial
from multiprocessing import Pool
from typing import List, Dict, Callable
from argparse import ArgumentParser, Namespace

sys.path.append(os.path.join(os.path.dirname(os.getcwd()), ".."))

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from AI.src.preprocessing import VideoPreprocessor
from AI.src.data.dataset import VideoFolderDataset
from AI.src.data.dataloader import DefaultDataLoader


def _custom_collate_fn(batch) -> List[str]:
    paths: List[str] = list(map(lambda item: item[0], batch))
    return paths


def _is_labeled(fpath: str, ds_name: str) -> bool:
    flag: bool = False
    path_components: List[str] = fpath.split(os.sep)

    if path_components[path_components.index(ds_name) + 1] == "labeled":
        flag = not flag
    return flag


def stage_one(args: Namespace, dl: DataLoader, ds_name: str) -> None:
    pool: Pool = Pool(processes=args.processes)

    for fpaths in tqdm(dl, total=len(dl), desc=f"Dataset: {ds_name}", colour="red"):
        if args.device == "both":
            cpu_ratio: int = int(args .cpu_ratio * len(fpaths))
            devices: List[str] = list(map(lambda i: "cpu" if i < cpu_ratio else "cuda", range(len(fpaths))))
        else:
            devices: List[str] = [args.device] * len(fpaths)

        is_labeleds: List[bool] = list(map(_is_labeled, *(fpaths, [ds_name] * len(fpaths))))

        preprocessors: List[VideoPreprocessor] = list(map(
            lambda fpath, device: VideoPreprocessor(
                fpath,
                args.save_root,
                ds_name,
                device,
                32,
                30
            ), *(fpaths, devices))
        )

        iterables = [(preprocessor, label_status, args.run_async) for preprocessor, label_status in zip(preprocessors, is_labeleds)]
        pool.starmap(VideoPreprocessor.stage_one, iterables)

        if args.run_async:
            time.sleep(5)
    return None


def stage_two(args: Namespace, dl: DataLoader, ds_name: str) -> None:
    pool: Pool = Pool(processes=args.processes)

    for fpaths in tqdm(dl, total=len(dl), desc=f"Dataset: {ds_name}", colour="red"):
        print(fpaths, args.del_prev_result)
        if args.device == "both":
            cpu_ratio: int = int(args .cpu_ratio * len(fpaths))
            devices: List[str] = list(map(lambda i: "cpu" if i < cpu_ratio else "cuda", range(len(fpaths))))
        else:
            devices: List[str] = [args.device] * len(fpaths)

        preprocessors: List[VideoPreprocessor] = list(map(
            lambda fpath, device: VideoPreprocessor(
                fpath,
                args.save_root,
                ds_name,
                device,
                32,
                30
            ), *(fpaths, devices))
        )

        pool.map(partial(VideoPreprocessor.stage_two, del_prev_result=args.del_prev_result), preprocessors)


def main(args: Namespace) -> None:
    STAGES: Dict[str, Callable] = {
        "stage_one": stage_one,
        "stage_two": stage_two
    }

    ds_name: str = pathlib.Path(args.root).name
    ds: Dataset = VideoFolderDataset(args.root, "v6")
    dl: DataLoader = DefaultDataLoader(ds, args.batch_size, None, collate_fn=_custom_collate_fn, drop_last=False)
    STAGES[args.fn_name](args, dl, ds_name)
    return None


if __name__ == "__main__":
    argument_parser = ArgumentParser()
    argument_parser.add_argument("--device",
                                 default="cpu",
                                 type=str,
                                 help="Device for preprocessing video with ffmpeg"
                                 )
    argument_parser.add_argument("--cpu_ratio",
                                 default=.5,
                                 type=float,
                                 help="Ratio b/t the utilization of cpu and gpu if device is both"
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

    argument_parser.add_argument("--fn_name",
                                 type=str,
                                 help="Which preprocessing stage to run")

    # Stage 1 only
    argument_parser.add_argument("--run_async",
                                 default=False,
                                 type=lambda x: (str(x).lower() == "true"),
                                 help="Run ffmpeg in an async manner")

    # Take effect from stage 2
    argument_parser.add_argument("--del_prev_result",
                                 default=False,
                                 type=bool,
                                 help="Delete previous stage result")
    parsed_args = argument_parser.parse_args()
    main(parsed_args)
