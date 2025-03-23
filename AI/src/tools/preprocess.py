"""Script for preprocessing train video."""
import os
import sys
import glob
import time
import shutil
import pathlib
import inspect

from functools import partial
from multiprocessing import Pool
from argparse import ArgumentParser, Namespace
from typing import List, Dict, Callable, Set, Any
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), ".."))

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# Workaround for not installing src as a package
from AI.src.preprocessing import VideoPreprocessor
from AI.src.data.dataset import VideoFolderDataset
from AI.src.data.dataloader import DefaultDataLoader


# Workaround for ffmpeg in win32
if sys.platform == "win32":
    print("Initializing DLL path for Windows")
    for path in os.environ.get("Path", "").split(";"):
        if os.path.exists(path):
            os.add_dll_directory(path)


def _custom_collate_fn(batch) -> List[str]:
    paths: List[str] = list(map(lambda item: item[0], batch))
    return paths


def stage_one(args: Namespace, dl: DataLoader, ds_name: str) -> None:
    pool: Pool = Pool(processes=args.processes)

    for fpaths in tqdm(dl, total=len(dl), desc=f"Dataset: {ds_name}", colour="cyan"):
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

        pool.map(partial(VideoPreprocessor.stage_one, run_async=args.run_async), preprocessors)

        if args.run_async:
            time.sleep(args.wait_time)
    return None


def stage_two(args: Namespace, dl: DataLoader, ds_name: str) -> None:
    pool: Pool = Pool(processes=args.processes)

    for fpaths in tqdm(dl, total=len(dl), desc=f"Dataset: {ds_name}", colour="red"):
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

        # Due to insufficient RAM
        is_labeled_processors: List[VideoPreprocessor] = [preprocessor for preprocessor in preprocessors if preprocessor.is_label]
        not_is_labeled_processors: List[VideoPreprocessor] = [preprocessor for preprocessor in preprocessors if not preprocessor.is_label]

        if len(is_labeled_processors) > 0:
            pool.map(partial(VideoPreprocessor.stage_two, del_prev_result=args.del_prev_result), is_labeled_processors)

        if len(not_is_labeled_processors) > 0:
            pool.map(partial(VideoPreprocessor.stage_two, del_prev_result=args.del_prev_result), not_is_labeled_processors)
    return None


def stage_three(args: Namespace, dl: DataLoader, ds_name: str) -> None:
    """
    Move all other file from original to save_root
    """
    ds_name_idx = args.root.split(os.sep).index(ds_name)

    save_root: List[str] = args.root.split(os.sep)
    save_root.insert(ds_name_idx, args.save_root)
    save_root: str = f"{os.sep}".join(save_root)

    objs_in_save_root: Set[str] = set([path.replace(save_root, "") for path in
                                       glob.glob(f"{save_root}/**", recursive=True, include_hidden=True)])

    objs_in_original_root: Set[str] = set([
        path.replace(args.root, "").replace(args.vid_ext, "pt")
        for path in glob.glob(f"{args.root}/**", recursive=True, include_hidden=True)
    ])

    for f in objs_in_original_root.difference(objs_in_save_root):
        original_path = f"{args.root}{f}"
        save_path = f"{save_root}{f}"
        shutil.copy(original_path, save_path)
    return None


def main(args: Namespace) -> None:
    STAGES: Dict[str, Callable] = {
        "stage_one": stage_one,
        "stage_two": stage_two,
        "stage_three": stage_three
    }
    callable_paras = set(inspect.signature(STAGES["stage_one"]).parameters.keys())

    ds_name: str = pathlib.Path(args.root).name
    ds: Dataset = VideoFolderDataset(args.root, "v6")
    dl: DataLoader = DefaultDataLoader(ds, args.batch_size, None, collate_fn=_custom_collate_fn, drop_last=False)

    kwargs: Dict[str, Any] = {"args": args, "dl": dl, "ds_name": ds_name}
    kwargs = {k: v for k, v in kwargs.items() if k in callable_paras}
    STAGES[args.fn_name](**kwargs)
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

    argument_parser.add_argument("--wait_time",
                                 default=20,
                                 type=int,
                                 help="Waiting time when running async manner")

    # Take effect from stage 2
    argument_parser.add_argument("--del_prev_result",
                                 default=False,
                                 type=lambda x: (str(x).lower() == "true"),
                                 help="Delete previous stage result")

    # Take effect from stage 3
    argument_parser.add_argument("--vid_ext",
                                 type=str,
                                 help="Video extension")

    parsed_args = argument_parser.parse_args()
    main(parsed_args)
