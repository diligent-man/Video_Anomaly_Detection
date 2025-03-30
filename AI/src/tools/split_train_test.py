import os
import copy
import shutil

from glob import glob
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Tuple


import pandas as pd

from tqdm import tqdm
from sklearn.model_selection import train_test_split


def _split_train_val(obj_lst: List[str],
                     ds_type: str,
                     train_ratio: float | int,
                     seed: int = 12345,
                     class_names: List[str] = None
                     ) -> Tuple[List[str], List[str]] | \
                          Tuple[List[str], List[str], List[str], List[str]]:
    assert ds_type in ("anomaly", "normal"), ValueError("Dataset type should be anomaly/ normal")
    if ds_type == "anomaly":
        assert class_names is not None, ValueError("Class list should be provided when dataset type is anomaly")

    if ds_type == "anomaly":
        x_train, x_val, y_train, y_val = train_test_split(
            obj_lst,
            class_names,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True,
            stratify=class_names
        )
        return x_train, x_val, y_train, y_val
    else:
        train, val = train_test_split(
            obj_lst,
            train_size=train_ratio,
            random_state=seed,
            shuffle=True
        )
        return train, val


def _get_fname(fpath: str | Path) -> str:
    fpath: Path = Path(fpath)
    fname: str = fpath.name.replace(fpath.suffix, "")
    return fname


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


def _move_obj(obj_lst: List[str],
              spath: str,
              ds_name: str,
              ds_phase: str,
              ds_type: str,
              counter: Dict[str, int],
              label: pd.DataFrame = None,
              class_names: List[str] = None
              ) -> None:
    """
    :param obj_lst: List of string of fpath
    :param ds_name: dataset name
    :param spath: save path dir which includes train/ val/ test finally
    :param ds_phase: "train"/ "val"/ "test"
    :param ds_type: "anomaly"/ "normal"
    :param counter: counter for increment file naming
    :param label: associated annotation for update name
    :param class_names: list of associated class name to vide (only provided for "anomaly")
    :return: Move file from obj_lst to spath based on ds_type & phase
    """
    desc: str = f"Moving {ds_name}/{ds_type} for {ds_phase}"
    for i, obj in tqdm(enumerate(obj_lst), colour="cyan", desc=desc, total=len(obj_lst)):
        obj: Path = Path(obj)
        counter_key: str = f"{ds_phase}_{ds_name}_{ds_phase}_{ds_type}"
        save_path: str = os.path.join(spath, ds_phase, ds_type)

        if class_names is not None:
            save_path = os.path.join(save_path, class_names[i])

        save_name: str = f"{ds_name}_{counter[counter_key]:06}{obj.suffix}"

        if not os.path.exists(save_path):
            os.makedirs(save_path, exist_ok=True)

        if not os.path.exists(os.path.join(save_path, save_name)):
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
        counter[f"{ds_phase}_{ds_name}_{ds_phase}_{ds_type}"] += 1
    return None


def unlabeled_to_train_val(ds_paths: List[str],
                           train_ratios: Dict[str, float],
                           spath: str,
                           counter: Dict[str, int],
                           seed: int = 12345
                           ):
    """
    Split and move unlabeled data to train & val in save path
    """
    os.makedirs(os.path.join(spath, "train"), exist_ok=True)
    os.makedirs(os.path.join(spath, "val"), exist_ok=True)

    ds_paths: List[str] = [os.path.join(path, "unlabeled") for path in ds_paths if "unlabeled" in os.listdir(path)]

    for ds_path in ds_paths:
        ds_name: str = Path(ds_path).parent.name
        prompt = f"Dataset name: {ds_name}\n"

        for ds_type in ("anomaly", "normal"):
            obj_lst: List[str] = glob(os.path.join(ds_path, ds_type, "**"), recursive=True)
            obj_lst = [obj for obj in obj_lst if obj.endswith(".pt")]

            class_names: None | List[str] = None
            if ds_type == "anomaly":
                class_names = [Path(path).parent.name for path in obj_lst]

            train, val = _split_train_val(obj_lst, ds_type, "train", train_ratios[ds_name], seed, class_names)

            prompt += f"""\tTotal {ds_type} datapoints: {len(obj_lst)}
        Train: {len(train)}
        Val: {len(val)}
"""
            for ds, ds_phase in zip((train, val), ("train", "val")):
                _move_obj(
                    ds,
                    spath,
                    ds_name,
                    ds_phase,
                    ds_type,
                    counter,
                    None,
                    class_names
                )
        print(prompt)
        print()
        print()
    return None


def labeled_to_train_val_test(ds_paths: List[str],
                              train_ratios: Dict[str, float],
                              spath: str,
                              counter: Dict[str, int],
                              seed: int = 12345
                              ) -> None:
    """
    Split and move labeled data to train & val in save path
    """
    df_headers: List[str] = ["path", "start1", "end1", "start2", "end2"]

    if os.path.exists(os.path.join(spath, "test", "label.csv")):
        dst_label: pd.DataFrame = pd.read_csv(os.path.join(spath, "test", "label.csv"))
    else:
        dst_label: pd.DataFrame = pd.DataFrame.from_dict({header: [] for header in df_headers})

    ds_paths: List[str] = [os.path.join(path, "labeled") for path in ds_paths if "labeled" in os.listdir(path)]
    for ds_path in ds_paths:
        src_label: pd.DataFrame = pd.read_csv(
            os.path.join(ds_path, "label.csv"),
            header=None,
            names=df_headers
        )

        ds_name: str = Path(ds_path).parent.name
        prompt = f"Dataset name: {ds_name}\n"

        for ds_type in ("anomaly", "normal"):
            obj_lst: List[str] = glob(os.path.join(ds_path, ds_type, "**"), recursive=True)
            obj_lst = [obj for obj in obj_lst if obj.endswith(".pt")]

            if ds_type == "anomaly":
                class_names = [Path(path).parent.name for path in obj_lst]
                (
                    train, test,
                    train_class_names, test_class_names
                ) = _split_train_val(obj_lst, ds_type, train_ratios[ds_name], seed, class_names)

                class_names = [Path(path).parent.name for path in train_class_names]
                (
                    train, val,
                    train_class_names, val_class_names
                ) = _split_train_val(train, ds_type, train_ratios[ds_name], seed, class_names)

                for ds, ds_phase, class_names in zip(
                        (train, val, test),
                        ("train", "val", "test"),
                        (train_class_names, val_class_names, test_class_names)
                ):
                    _move_obj(ds, spath,
                              ds_name, ds_phase,
                              ds_type, counter,
                              src_label, class_names)
            else:
                train, test = _split_train_val(obj_lst, ds_type, train_ratios[ds_name], seed)
                train, val = _split_train_val(train, ds_type, train_ratios[ds_name], seed)

                for ds, ds_phase in zip(
                        (train, val, test),
                        ("train", "val", "test")
                ):
                    _move_obj(ds, spath,
                              ds_name, ds_phase,
                              ds_type, counter,
                              src_label)

            prompt += f"""\tTotal {ds_type} datapoints: {len(obj_lst)}
            Train: {len(train)}
            Val: {len(val)}
            Test: {len(test)}
"""
        print(src_label)
        dst_label = pd.concat((dst_label, src_label))
        print(prompt)
        print()
        print()
    dst_label.to_csv(os.path.join(spath, "test", "label.csv"), index=False)
    return None


def labeled_to_test(ds_paths: List[str],
                    spath: str,
                    counter: Dict[str, int]
                    ) -> None:
    """
    Split and move labeled data to test in save path
    """
    df_headers: List[str] = ["path", "start1", "end1", "start2", "end2"]

    if os.path.exists(os.path.join(spath, "test", "label.csv")):
        dst_label: pd.DataFrame = pd.read_csv(os.path.join(spath, "test", "label.csv"))
    else:
        dst_label: pd.DataFrame = pd.DataFrame.from_dict({header: [] for header in df_headers})

    ds_paths: List[str] = [os.path.join(path, "labeled") for path in ds_paths if "labeled" in os.listdir(path)]
    for ds_path in ds_paths:
        src_label: pd.DataFrame = pd.read_csv(
            os.path.join(ds_path, "label.csv"),
            header=None,
            names=df_headers
        )

        ds_name: str = Path(ds_path).parent.name
        prompt = f"Dataset name: {ds_name}\n"

        for ds_type in ("anomaly", "normal"):
            obj_lst: List[str] = glob(os.path.join(ds_path, ds_type, "**"), recursive=True)
            obj_lst = [obj for obj in obj_lst if obj.endswith(".pt")]

            class_names: None | List[str] = None
            if ds_type == "anomaly":
                class_names = [Path(path).parent.name for path in obj_lst]

            prompt += f"""\tTotal {ds_type} datapoints: {len(obj_lst)}
        Test: {len(obj_lst)}
"""
            _move_obj(
                obj_lst,
                spath,
                ds_name,
                "test",
                ds_type,
                counter,
                src_label,
                class_names
            )
        dst_label = pd.concat((dst_label, src_label))
        print(prompt)
        print()
        print()
    dst_label.to_csv(os.path.join(spath, "test", "label.csv"), index=False)
    return None


def main() -> None:
    spath: str = r"D:\Dataset\VAD\final_ds"
    counter: Dict[str, int] = defaultdict(int)  # Key format: f"{ds_phase}_{ds_name}_{ds_phase}_{ds_type}"

    # unlabeled_to_train_val(
    #     [
    #         r"D:\Dataset\VAD\out\crawled_data",
    #         r"D:\Dataset\VAD\out\IITB",
    #         r"D:\Dataset\VAD\out\UCF"
    #     ],
    #     {
    #         "crawled_data": 0.85,
    #         "IITB": 0.85,
    #         "UCF": 0.85
    #     },
    #     copy.deepcopy(spath),
    #     counter
    # )

    labeled_to_train_val_test(
        [
            # r"D:\Dataset\VAD\out\UBI-FIGHT"
            r"D:\Dataset\VAD\out\UCF"
        ],
        {
            "UBI-FIGHT": 0.8,
            "UCF": 0.8,
        },
        copy.deepcopy(spath),
        counter
    )

    # labeled_to_test(
    #     [
    #         r"D:\Dataset\VAD\out\UCF",
    #     ],
    #     copy.deepcopy(spath),
    #     counter
    # )
    return None


if __name__ == '__main__':
    main()
