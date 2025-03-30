import os
import shutil

from glob import glob
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple


import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split


__all__ = [
    "unlabeled_to_train_val",
    "labeled_to_test",
    "labeled_to_train_val_test"
]


def unlabeled_to_train_val(ds_paths: List[str],
                           train_ratios: Dict[str, float],
                           spath: str,
                           counter: Dict[str, int],
                           seed: int = 12345
                           ) -> None:
    """
    Split and move unlabeled data to train & val in save path. All files are in pytorch save format
    """
    os.makedirs(os.path.join(spath, "train"), exist_ok=True)
    os.makedirs(os.path.join(spath, "val"), exist_ok=True)

    ds_cond: str = "unlabeled"
    ds_paths: List[str] = [os.path.join(path, ds_cond) for path in ds_paths if _ds_filter(path, ds_cond)]

    for ds_path in ds_paths:
        sample_traker: Dict[str, int] = defaultdict(int)
        ds_name: str = Path(ds_path).parent.name
        prompt = f"Dataset name: {ds_name}\n"

        if os.path.exists(os.path.join(spath, f"{ds_name}_log.txt")):
            os.remove(os.path.join(spath, f"{ds_name}_log.txt"))
        log_writer = open(os.path.join(spath, f"{ds_cond}_{ds_name}_log.txt"), mode="w")

        for ds_type in ("anomaly", "normal"):
            obj_lst: List[str] = [
                obj for obj in glob(os.path.join(ds_path, ds_type, "**"), recursive=True)
                if obj.endswith((".pt", ".pth"))
            ]

            if ds_type == "anomaly":
                cls_names: List[str] = [Path(path).parent.name for path in obj_lst]
                (
                    train, val,
                    train_cls_names, val_cls_names
                ) = _split_train_val(obj_lst, ds_type, train_ratios[ds_name], seed, cls_names)

                for obj_lst, ds_phase, cls_names in zip(
                        (train, val),
                        ("train", "val"),
                        (train_cls_names, val_cls_names)
                ):
                    _move_obj(obj_lst, spath, ds_phase, ds_name, ds_type, counter, None, cls_names, log_writer)
                    sample_traker = _track_sample(sample_traker, f"{ds_phase}_{ds_type}", len(obj_lst))
            else:
                train, val = _split_train_val(obj_lst, ds_type, train_ratios[ds_name], seed)
                for obj_lst, ds_phase in zip((train, val), ("train", "val"),):
                    _move_obj(obj_lst, spath, ds_phase, ds_name, ds_type, counter, log_writer=log_writer)
                    sample_traker = _track_sample(sample_traker, f"{ds_phase}_{ds_type}", len(obj_lst))

            sample_traker[ds_type] += len(train) + len(val)
            prompt += f"""\tTotal {ds_type} datapoints: {sample_traker[ds_type]}
        Train: {sample_traker[f"train_{ds_type}"]}
        Val: {sample_traker[f"val_{ds_type}"]}
"""
        print(prompt)
        print()
        log_writer.close()
    return None


def labeled_to_train_val_test(ds_paths: List[str],
                              train_ratios: Dict[str, float],
                              spath: str,
                              counter: Dict[str, int],
                              seed: int = 12345
                              ) -> None:
    """
    Split and move labeled data to train/ val/ test in save path
    """
    ds_cond: str = "labeled"
    dst_label = _get_label(spath)
    ds_paths: List[str] = [os.path.join(path, ds_cond) for path in ds_paths if ds_cond in os.listdir(path)]

    for ds_path in ds_paths:
        ds_name: str = Path(ds_path).parent.name
        prompt = f"Dataset name: {ds_name}\n"
        sample_traker: Dict[str, int] = defaultdict(int)

        src_label: pd.DataFrame = pd.read_csv(
            os.path.join(ds_path, "label.csv"),
            header=None,
            names=dst_label.columns
        )

        if os.path.exists(os.path.join(spath, f"{ds_name}_log.txt")):
            os.remove(os.path.join(spath, f"{ds_name}_log.txt"))
        log_writer = open(os.path.join(spath, f"{ds_cond}_{ds_name}_log.txt"), mode="w")

        for ds_type in ("anomaly", "normal"):
            obj_lst: List[str] = [
                obj for obj in glob(os.path.join(ds_path, ds_type, "**"), recursive=True)
                if obj.endswith((".pt", ".pth"))
            ]

            if ds_type == "anomaly":
                cls_names: List[str] = [Path(path).parent.name for path in obj_lst]
                (
                    train, test,
                    train_cls_names, test_cls_names
                ) = _split_train_val(obj_lst, ds_type, train_ratios[ds_name], seed, cls_names)

                (
                    train, val,
                    train_cls_names, val_cls_names
                ) = _split_train_val(train, ds_type, train_ratios[ds_name], seed, train_cls_names)

                for obj_lst, ds_phase, class_names in zip(
                        (train, val, test),
                        ("train", "val", "test"),
                        (train_cls_names, val_cls_names, test_cls_names)
                ):
                    _move_obj(obj_lst, spath, ds_phase, ds_name, ds_type, counter, src_label, cls_names, log_writer)
                    sample_traker = _track_sample(sample_traker, f"{ds_phase}_{ds_type}", len(obj_lst))
            else:
                train, test = _split_train_val(obj_lst, ds_type, train_ratios[ds_name], seed)
                train, val = _split_train_val(train, ds_type, train_ratios[ds_name], seed)

                for obj_lst, ds_phase in zip((train, val, test), ("train", "val", "test")):
                    _move_obj(obj_lst, spath, ds_phase, ds_name, ds_type, counter, src_label, None, log_writer)
                    sample_traker = _track_sample(sample_traker, f"{ds_phase}_{ds_type}", len(obj_lst))

            sample_traker[ds_type] += len(train) + len(val) + len(test)
            prompt += f"""\tTotal {ds_type} datapoints: {sample_traker[ds_type]}
        Train: {sample_traker[f"train_{ds_type}"]}
        Val: {sample_traker[f"val_{ds_type}"]}
        Test: {sample_traker[f"test_{ds_type}"]}
"""
        dst_label = pd.concat((dst_label, src_label))
        dst_label = dst_label.sort_values(by=dst_label.columns[0])
        dst_label.to_csv(os.path.join(spath, "test", "label.csv"), index=False, header=False)
        log_writer.close()

        print(prompt)
        print()
    return None


def labeled_to_test(ds_paths: List[str],
                    spath: str,
                    counter: Dict[str, int],
                    seed: int = 12345
                    ) -> None:
    ds_cond: str = "labeled"
    dst_label = _get_label(spath)
    ds_paths: List[str] = [os.path.join(path, ds_cond) for path in ds_paths if ds_cond in os.listdir(path)]

    for ds_path in ds_paths:
        ds_name: str = Path(ds_path).parent.name
        prompt = f"Dataset name: {ds_name}\n"
        sample_traker: Dict[str, int] = defaultdict(int)

        src_label: pd.DataFrame = pd.read_csv(
            os.path.join(ds_path, "label.csv"),
            header=None,
            names=dst_label.columns
        )

        if os.path.exists(os.path.join(spath, f"{ds_name}_log.txt")):
            os.remove(os.path.join(spath, f"{ds_name}_log.txt"))
        log_writer = open(os.path.join(spath, f"{ds_cond}_{ds_name}_log.txt"), mode="w")

        for ds_type in ("anomaly", "normal"):
            obj_lst: List[str] = [
                obj for obj in glob(os.path.join(ds_path, ds_type, "**"), recursive=True)
                if obj.endswith((".pt", ".pth"))
            ]

            if ds_type == "anomaly":
                cls_names: List[str] = [Path(path).parent.name for path in obj_lst]
                (
                    train, test,
                    train_cls_names, test_cls_names
                ) = _split_train_val(obj_lst, ds_type, 0.5, seed, cls_names)
                test_cls_names: List[str] = [*train_cls_names, *test_cls_names]
            else:
                train, test = _split_train_val(obj_lst, ds_type, .5, seed)
                test_cls_names: None = None

            test: List[str] = [*train, *test]
            _move_obj(obj_lst, spath, "test", ds_name, ds_type, counter, src_label, test_cls_names, log_writer)
            sample_traker = _track_sample(sample_traker, f"test_{ds_type}", len(test))

            sample_traker[ds_type] += len(test)
            prompt += f"""\tTotal {ds_type} datapoints: {sample_traker[ds_type]}
        Test: {sample_traker[f"test_{ds_type}"]}
"""
        dst_label = pd.concat((dst_label, src_label))
        dst_label = dst_label.sort_values(by=dst_label.columns[0])
        dst_label.to_csv(os.path.join(spath, "test", "label.csv"), index=False, header=False)
        log_writer.close()

        print(prompt)
        print()
    return None


########################################################################################################################
def _get_fname(fpath: str | Path) -> str:
    fpath: Path = Path(fpath)
    fname: str = fpath.name.replace(fpath.suffix, "")
    return fname


def _ds_filter(path: str, cond: str) -> bool:
    return True if cond in os.listdir(Path(path)) else False


def _get_label(spath: str) -> pd.DataFrame:
    """
    :param spath: root path to final dataset.
    :return: label file for test set. This is true for VAD problem
    """
    headers: List[str] = ["path", "start1", "end1", "start2", "end2"]

    if os.path.exists(os.path.join(spath, "test", "label.csv")):
        df: pd.DataFrame = pd.read_csv(os.path.join(spath, "test", "label.csv"), header=None, names=headers)
    else:
        df: pd.DataFrame = pd.DataFrame.from_dict({header: [] for header in headers})
    return df


def _track_sample(tracker: Dict[str, int], k: str, val: int):
    tracker[k] += val
    return tracker


def _rename_label(label_fpath: str, old_fname: str, new_fname: str, ext: str) -> str:
    """
    :param label_fpath: fpath in label file
    :param old_fname: obj name in src
    :param new_fname: obj name in dst
    :param ext: src obj extension (after run stage 3 in preprocessing -> ".pt")
    :return: renamed fpath in label
    """
    old_fname: str = old_fname.replace(ext, "")
    label_fname: str = _get_fname(label_fpath)

    if label_fname == old_fname:
        label_fpath: str = os.path.join(Path(label_fpath).parent, new_fname)
    return label_fpath


def _split_train_val(obj_lst: List[str],
                     ds_type: str,
                     train_ratio: float | int,
                     seed: int = 12345,
                     cls_names: List[str] = None,
                     return_cls: bool = True
                     ) -> Tuple[List[str], List[str]] | \
                          Tuple[List[str], List[str], List[str], List[str]]:
    assert ds_type in ("anomaly", "normal"), ValueError("Dataset type should be anomaly/ normal")

    if ds_type == "anomaly":
        assert cls_names is not None, ValueError("Class list should be provided when dataset type is anomaly")

    if ds_type == "anomaly":
        x_train, x_val, y_train, y_val = train_test_split(
            obj_lst,
            cls_names,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True,
            stratify=cls_names
        )
        if return_cls:
            return x_train, x_val, y_train, y_val
        else:
            return x_train, x_val
    else:
        train, val = train_test_split(
            obj_lst,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True
        )
        return train, val


def _move_obj(obj_lst: List[str],
              spath: str,
              ds_phase: str,
              ds_name: str,
              ds_type: str,
              counter: Dict[str, int],
              label: pd.DataFrame = None,
              cls_names: List[str] = None,
              log_writer=None
              ) -> None:
    """
    :param obj_lst: List of string of fpath
    :param spath: save path dir which includes train/ val/ test finally
    :param ds_phase: "train"/ "val"/ "test"
    :param ds_name: dataset name
    :param ds_type: "anomaly"/ "normal"
    :param counter: counter for increment file naming
    :param label: associated annotation for update name
    :param cls_names: list of associated class name to vide (only provided for "anomaly")
    :param log_writer: log writer for tracking src and dst file
    :return: Move file from obj_lst to spath based on ds_type & phase
    """
    desc: str = f"Moving {ds_name}/{ds_type} for {ds_phase}"

    for i, obj in tqdm(enumerate(obj_lst), colour="cyan", desc=desc, total=len(obj_lst)):
        obj: Path = Path(obj)

        counter_key: str = f"{ds_phase}_{ds_name}_{ds_type}"
        save_path: str = os.path.join(spath, ds_phase, ds_type)

        if cls_names is not None:
            save_path = os.path.join(save_path, cls_names[i])

        save_name: str = f"{ds_name}_{counter[counter_key]:06}{obj.suffix}"

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        if not os.path.exists(os.path.join(save_path, save_name)):
            if log_writer is not None:
                ds_name_idx: int = str(obj).split(os.sep).index(ds_name)

                log_writer.write(
                    f"{f'{os.sep}'.join(str(obj).split(os.sep)[ds_name_idx+1:])}, "
                    f"{os.path.join(save_path.replace(spath, ''), save_name)}\n"
                )

            shutil.copy(
                obj,
                os.path.join(save_path, save_name)
            )

        if label is not None:
            if ds_phase == "test":
                label.path = label.path.apply(_rename_label, args=(obj.name, save_name, obj.suffix))
            else:
                label.drop(label[label.path.apply(_get_fname) == obj.name.replace(obj.suffix, "")].index, inplace=True)
                label.reset_index(drop=True, inplace=True)
        counter[f"{ds_phase}_{ds_name}_{ds_type}"] += 1
    return None
