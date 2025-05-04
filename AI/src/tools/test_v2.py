"""
Temp functional test code
"""
import os
import gc
import sys
import multiprocessing

from typing import Tuple, Mapping, Any, List, Dict
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), "../../"))


import torch
from tqdm import tqdm
from torch.nn import Module


from AI.src.utils import Logger
from AI.src.data.dataloader import DefaultDataLoader
from AI.src.utils.inference_ops import dispatch_infer
from AI.src.modeling.architectures import build_model
from AI.src.data.dataset import VADFrameLevelTestDataset
from AI.src.utils import DotDict, load_config, load_weights, VideoCache


global starter


def main() -> None:
    multiprocessing.set_start_method("spawn")

    device = "cuda"
    T_max: int = 30
    overlap_ratio: float = 0.5

    batch_thres = [1000, 2000, 3000, 4500, 6000, 10000, 15000, 20000, torch.inf]
    batch_worker = [16] * 4 + [14] * 2 + [12] * 3
    video_cache = VideoCache(batch_thres, batch_worker)

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

        for idx, inp, label in tqdm(dl, total=len(dl)):
            idx: torch.Tensor
            inp: Tuple[str]

            idx: int = idx.item()
            inp: str = inp[0]

            video_cache.cache(label.squeeze(0).shape[0], ["inp", "label", "idx"], [inp, label, idx])

            cache: Dict[str, Any] | None = video_cache.get_cache()

            if cache is not None:
                print("Batch:", cache["batch_worker"])
                result: Tuple[List[float], List[int]] = dispatch_infer(cache, model, device, T_max, overlap_ratio)

                for i in range(len(result)):
                    log_info = f"{result[i][0]},{result[i][1]},{cache['idx'][i]}\n"
                    logger.write(pred_result, log_info, "a")
                gc.collect()
                torch.cuda.empty_cache()

        # Remaining
        print("Run leftovers")
        for cache in video_cache.get_remains():
            print("Batch:", cache["batch_worker"])
            result: Tuple[List[float], List[int]] = dispatch_infer(cache, model, device, T_max, overlap_ratio)

            for i in range(len(result)):
                log_info = f"{result[i][0]},{result[i][1]},{cache['idx'][i]}\n"
                logger.write(pred_result, log_info, "a")
            gc.collect()
            torch.cuda.empty_cache()
    return None


if __name__ == '__main__':
    main()
