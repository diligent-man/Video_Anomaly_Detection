"""
Temp test
250
model 3: RuntimeError: Input type (torch.FloatTensor) and weight type (torch.cuda.FloatTensor) should be the same or input should be a MKLDNN tensor and weight is a dense tensor

"""
import os
import gc
import sys
import time
import multiprocessing

from multiprocessing import Pool
from collections import defaultdict
from typing import Tuple, Mapping, Any, List, Dict
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "../../"))


import torch
from tqdm import tqdm
from torch.nn import Module


from AI.src.utils import Logger
from AI.src.data.dataloader import DefaultDataLoader
from AI.src.modeling.architectures import build_model
from AI.src.data.dataset import VADFrameLevelTestDataset
from AI.src.utils.inference_ops import infer_for_test_v2
from AI.src.utils import DotDict, load_config, load_weights


global starter


class VideoCache(object):
    __cache: Dict[int, Dict[str, List[Any]]]

    __cached_k: List[str] = None
    __batch_worker: List[int] = [8] * 5
    __batch_thres: List[int] = [0, 3000, 6000, 10000, 20000, torch.inf]

    def __init__(self,
                 batch_thres: List[int] = None,
                 batch_worker: List[int] = None,
                 ) -> None:
        if batch_thres is not None and batch_worker is not None:
            assert len(batch_worker) == len(batch_thres), ValueError("num worker should equal num threshold")
            batch_thres = [0] + batch_thres

            self.__batch_thres = batch_thres
            self.__batch_worker = batch_worker

        self.__cache = {k: defaultdict(list) for k in self.__batch_thres[1:]}

    def cache(self, video_len: int, k_lst: List[str], v_lst: List[Any]) -> None:
        if self.__cached_k is None:
            self.__cached_k = k_lst

        for i in range(len(self.__batch_thres)-1):
            if self.__batch_thres[i] <= video_len <= self.__batch_thres[i+1]:
                for k, v in zip(k_lst, v_lst):
                    self.__cache[self.__batch_thres[i+1]][k].append(v)

    def get_cache(self) -> Dict[str, Any] | None:
        result: Dict[str, Any] | None = None

        for i in range(1, len(self.__batch_thres)):
            first_k = self.__cached_k[0]

            if len(self.__cache[self.__batch_thres[i]][first_k]) == self.__batch_worker[i-1]:
                result = self.__cache[self.__batch_thres[i]]
                result["batch_worker"] = self.__batch_worker[i-1]
                self.__cache[self.__batch_thres[i]] = defaultdict(list)
                break
        return result

    def get_remains(self) -> Dict[str, Any]:
        for i in range(1, len(self.__batch_thres)):
            first_k = self.__cached_k[0]

            if len(self.__cache[self.__batch_thres[i]][first_k]) > 0:
                result: Dict[str, Any] = self.__cache[self.__batch_thres[i]]
                result["batch_worker"] = self.__batch_worker[i-1]
                yield result


def init_proc(shared_val: multiprocessing.Value, batch_worker: int) -> None:
    # ref: https://stackoverflow.com/a/70449572
    global starter
    starter = shared_val
    with starter.get_lock():
        if batch_worker == 15:
            time.sleep(6)
        elif batch_worker == 14:
            time.sleep(4)
        else:
            time.sleep(0)


def dispatch_infer(cache: Dict[str, Any],
                   model: Module,
                   device: str,
                   T_max: int,
                   overlap_ratio: float) -> Tuple[List[float], List[int]]:
    with Pool(min(32, cache["batch_worker"]), init_proc, [multiprocessing.Value("d"), cache["batch_worker"]]) as pool:
        result: Tuple[List[float], List[int]] = pool.starmap(
            infer_for_test_v2,
            zip(cache["inp"],
                cache["label"],
                [model] * len(cache["inp"]),
                [device] * len(cache["inp"]),
                [T_max] * len(cache["inp"]),
                [overlap_ratio] * len(cache["inp"])
                )
        )
        return result


def main() -> None:
    multiprocessing.set_start_method('spawn')

    device = "cuda"
    T_max: int = 30
    overlap_ratio: float = 0.5

    for pred_result, config, weight in zip(
        [
            "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_2d/v1/log/pred_result.txt",
            # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_2d/v2/log/pred_result.txt",
            # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_2d/v3/log/pred_result.txt",
            # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_3d/v1/log/pred_result.txt",
            "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_3d/v2/log/pred_result.txt",
            # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_3d/v3/log/pred_result.txt",
            # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_mixed/v1/log/pred_result.txt",
            # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_mixed/v2/log/pred_result.txt",
            "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_mixed/v3/log/pred_result.txt"
         ],
        [
            "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_2d/v1/Mlflow/995263845449942640/d4e6cc59499a4abc90cf6410eb9aef25/artifacts/config.json",
            # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_2d/v2/Mlflow/371379892464714510/ca8300cd05d4495db1070b9d9ea987d8/artifacts/config.json",
            # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_2d/v3/Mlflow/118746316556651065/dd5c8e46f4274cd481ea91465b591ea9/artifacts/config.json",
            # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_3d/v1/Mlflow/890005033899140195/6653ca3fc5da40cbaf0101124bac5d70/artifacts/config.json",
            "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_3d/v2/Mlflow/341932603411297071/49d947ac8c4c43758a91dbbfb4b1505c/artifacts/config.json",
            # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_3d/v3/Mlflow/510859847621159024/b814d78b360d4a09a29847d7f65d44e6/artifacts/config.json",
            # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_mixed/v1/Mlflow/401041024666025539/619caa95c3a4476b89d5d91b0fc59cc1/artifacts/config.json",
            # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_mixed/v2/Mlflow/825420379933310267/7b53e9cc5e3c4835b086a18e76d0063d/artifacts/config.json",
            "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_mixed/v3/Mlflow/991378426196781897/aaff4465b324425b91421ee50a683603/artifacts/config.json"
        ],
        [
            "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_2d/v1/Mlflow/995263845449942640/d4e6cc59499a4abc90cf6410eb9aef25/artifacts/ckpt/best_epoch18_step4067.pt",
            # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_2d/v2/Mlflow/371379892464714510/ca8300cd05d4495db1070b9d9ea987d8/artifacts/ckpt/best_epoch13_step2937.pt",
            # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_2d/v3/Mlflow/118746316556651065/dd5c8e46f4274cd481ea91465b591ea9/artifacts/ckpt/best_epoch13_step2937.pt",
            # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_3d/v1/Mlflow/890005033899140195/6653ca3fc5da40cbaf0101124bac5d70/artifacts/ckpt/best_epoch14_step3163.pt",
            "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_3d/v2/Mlflow/341932603411297071/49d947ac8c4c43758a91dbbfb4b1505c/artifacts/ckpt/best_epoch20_step4519.pt",
            # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_3d/v3/Mlflow/510859847621159024/b814d78b360d4a09a29847d7f65d44e6/artifacts/ckpt/best_epoch20_step4519.pt",
            # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_mixed/v1/Mlflow/401041024666025539/619caa95c3a4476b89d5d91b0fc59cc1/artifacts/ckpt/best_epoch13_step2937.pt",
            # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_mixed/v2/Mlflow/825420379933310267/7b53e9cc5e3c4835b086a18e76d0063d/artifacts/ckpt/best_epoch9_step2033.pt",
            "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_mixed/v3/Mlflow/991378426196781897/aaff4465b324425b91421ee50a683603/artifacts/ckpt/best_epoch12_step2711.pt",
        ]
    ):
        config: str
        weight: str

        logger: Logger = Logger("test")
        config: DotDict = DotDict(load_config(config))
        weight: Mapping[str, Any] = load_weights(weight, weights_only=False)

        model: Module = build_model(config)
        model.load_state_dict(weight["model"]) if isinstance(weight["model"], dict) else model.load_state_dict(
            weight["model"].state_dict())
        model = model.to(device)

        dl = DefaultDataLoader(
            VADFrameLevelTestDataset(
                "/home/trong/Downloads/Dataset/VAD/final/test",
                "label.csv",
            ), num_workers=4, shuffle=False, multiprocessing_context="fork"
        )

        batch_thres = [1000, 2000, 3000, 4500, 6000, 10000, 15000, 20000, torch.inf]
        batch_worker = [16] * 4 + [15] * 2 + [14] * 3
        video_cache = VideoCache(batch_thres, batch_worker)

        for idx, inp, label in tqdm(dl, total=len(dl)):
            idx: torch.Tensor
            inp: Tuple[str]

            idx: int = idx.item()
            inp: str = inp[0]

            video_cache.cache(
                label.squeeze(0).shape[0],
                ["inp", "label", "idx"],
                [inp, label, idx]
            )

            cache: Dict[str, Any] | None = video_cache.get_cache()

            if cache is not None:
                print("Batch:", cache["batch_worker"])
                if cache["batch_worker"] != 14:
                    continue

                result: Tuple[List[float], List[int]] = dispatch_infer(cache, model, device, T_max, overlap_ratio)

                for i in range(len(result)):
                    log_info = f"{result[i][0]},{result[i][1]},{cache['idx'][i]}\n"
                    logger.write(pred_result, log_info, "a")
                gc.collect()
                torch.cuda.empty_cache()
                exit()
        # Remaining
        for cache in video_cache.get_remains():
            result: Tuple[List[float], List[int]] = dispatch_infer(cache, model, device, T_max, overlap_ratio)

            for i in range(len(result)):
                log_info = f"{result[i][0]},{result[i][1]},{cache['idx'][i]}\n"
                logger.write(pred_result, log_info, "a")
            gc.collect()
            torch.cuda.empty_cache()
    return None


if __name__ == '__main__':
    main()
