from typing import Dict
from copy import deepcopy
from collections import defaultdict
from AI.src.preprocessing.split_train_test import *


def main() -> None:
    spath: str = "/home/trong/Downloads/Dataset/VAD/tmp_test/final/IITB"

    # Key format: f"{ds_phase}_{ds_name}_{ds_phase}_{ds_type}"
    counter: Dict[str, int] = defaultdict(int)

    # unlabeled_to_train_val(
    #     [
    #         "/home/trong/Downloads/Dataset/VAD/preprocessing/stage_2/train/crawled_data/",
    #         "/home/trong/Downloads/Dataset/VAD/preprocessing/stage_2/train/IITB/",
    #         "/home/trong/Downloads/Dataset/VAD/preprocessing/stage_2/train/UCF/",
    #     ],
    #     {
    #         "crawled_data": 0.85,
    #         "IITB": 0.85,
    #         "UCF": 0.85
    #     },
    #     deepcopy(spath),
    #     counter,
    #     overwrite_prev_log=True
    # )

    labeled_to_test(
        [
            "/home/trong/Downloads/Dataset/VAD/tmp_test/out/IITB"
        ],
        deepcopy(spath),
        counter,
        overwrite_prev_log=True
    )

    # labeled_to_train_val(
    #     [
    #         "/home/trong/Downloads/Dataset/VAD/preprocessing/stage_2/train/IITB",
    #         "/home/trong/Downloads/Dataset/VAD/preprocessing/stage_2/train/UBI-FIGHT",
    #     ],
    #     {
    #         "IITB": 0.8,
    #         "UBI-FIGHT": 0.8,
    #     },
    #     deepcopy(spath),
    #     counter,
    #     overwrite_prev_log=False
    # )
    return None


if __name__ == '__main__':
    main()
