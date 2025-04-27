"""
Temp test
"""
import gc
import multiprocessing

from tqdm import tqdm
from multiprocessing import Pool
from collections import defaultdict
from typing import Tuple, Mapping, Any, List


import torch
from torch.nn import Module
from torchvision.transforms import Compose, v2


from AI.src.utils import Logger
from AI.src.data.dataset import VADFrameLevelDataset
from AI.src.data.dataloader import DefaultDataLoader
from AI.src.utils.inference_ops import infer_for_test
from AI.src.modeling.architectures import build_model
from AI.src.utils import DotDict, load_config, load_weights


def main() -> None:
    multiprocessing.set_start_method('spawn')

    device = "cuda"
    T_max: int = 30
    overlap_ratio: float = 0.5

    logger: Logger = Logger("test")
    pred_result: str = "/home/trong/Downloads/Dataset/VAD/tmp_test/final/UCF/model_3_pred_result.txt"

    config: DotDict = DotDict(load_config(
        # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_2d/v1/Mlflow/995263845449942640/d4e6cc59499a4abc90cf6410eb9aef25/artifacts/config.json"
        # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_3d/v2/Mlflow/341932603411297071/49d947ac8c4c43758a91dbbfb4b1505c/artifacts/config.json"
        "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_mixed/v3/Mlflow/991378426196781897/aaff4465b324425b91421ee50a683603/artifacts/config.json"
    ))

    weight: Mapping[str, Any] = load_weights(
        # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_2d/v1/Mlflow/995263845449942640/d4e6cc59499a4abc90cf6410eb9aef25/artifacts/ckpt/best_epoch18_step4067.pt",
        # "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_3d/v2/Mlflow/341932603411297071/49d947ac8c4c43758a91dbbfb4b1505c/artifacts/ckpt/best_epoch20_step4519.pt",
        "/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/final_train_result/teacher/input_mixed/v3/Mlflow/991378426196781897/aaff4465b324425b91421ee50a683603/artifacts/ckpt/best_epoch12_step2711.pt",
        weights_only=False
    )

    model: Module = build_model(config)
    model.load_state_dict(weight["model"]) if isinstance(weight["model"], dict) else model.load_state_dict(
        weight["model"].state_dict())
    model = model.to(device)

    dl = DefaultDataLoader(
        VADFrameLevelDataset(
            "/home/trong/Downloads/Dataset/VAD/tmp_test/final/UCF/test",
            "label.csv",
            "v4",
            transform=Compose([v2.ToDtype(torch.float32, scale=True)])
        ), num_workers=1, shuffle=False, multiprocessing_context="fork"
    )

    # batch_thres = [0, 2500, 5000, 10000, 20000, 500000]
    # batch_worker = [16, 16, 16, 16, 12]

    batch_thres = [0, 1000, 2000, 3000]
    batch_worker = [8, 8, 8, 8]

    mp_inp = {i: defaultdict(list) for i in batch_thres[1:]}

    for i, (inp, label) in tqdm(enumerate(dl), total=len(dl)):
        for j in range(len(batch_thres)-1):
            if batch_thres[j] <= label.squeeze(0).shape[0] <= batch_thres[j+1]:
                mp_inp[batch_thres[j+1]]["inp"].append(inp)
                mp_inp[batch_thres[j+1]]["label"].append(label)
                mp_inp[batch_thres[j+1]]["idx"].append(i)

            if len(mp_inp[batch_thres[j+1]]["inp"]) == batch_worker[j] or i == len(dl)-1:
                with Pool(processes=min(32, batch_worker[j])) as pool:
                    result: Tuple[List[float], List[int]] = pool.starmap(
                        infer_for_test,
                        zip(mp_inp[batch_thres[j+1]]["inp"],
                            mp_inp[batch_thres[j+1]]["label"],
                            [model] * len(mp_inp[batch_thres[j+1]]["inp"]),
                            [device] * len(mp_inp[batch_thres[j+1]]["inp"]),
                            [T_max] * len(mp_inp[batch_thres[j+1]]["inp"]),
                            [overlap_ratio] * len(mp_inp[batch_thres[j+1]]["inp"]),
                            [True] * len(mp_inp[batch_thres[j + 1]]["inp"])
                            )
                    )

                    log_info = ""
                    for k in range(min(batch_worker[j], len(mp_inp[batch_thres[j+1]]["inp"]))):
                        log_info += f"{result[k][0]},{result[k][1]},{mp_inp[batch_thres[j+1]]['idx'][k]}\n"
                    logger.write(pred_result, log_info, "a")
                mp_inp[batch_thres[j+1]] = defaultdict(list)
                gc.collect()
                torch.cuda.empty_cache()
    return None


if __name__ == '__main__':
    main()
