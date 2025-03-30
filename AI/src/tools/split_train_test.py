from typing import Dict
from copy import deepcopy
from collections import defaultdict
from AI.src.preprocessing.split_train_test import *


def main() -> None:
    spath: str = r"D:\Dataset\VAD\final"

    # Key format: f"{ds_phase}_{ds_name}_{ds_phase}_{ds_type}"
    counter: Dict[str, int] = defaultdict(int)

    unlabeled_to_train_val(
        [
            r"D:\Dataset\VAD\preprocessed\crawled_data",
            # r"D:\Dataset\VAD\preprocessed\IITB",
            # r"D:\Dataset\VAD\preprocessed\UCF",
        ],
        {
            "crawled_data": 0.85,
            "IITB": 0.85,
            "UCF": 0.85
        },
        deepcopy(spath),
        counter
    )

    labeled_to_train_val_test(
        [
            r"D:\Dataset\VAD\preprocessed\IITB",
            # r"D:\Dataset\VAD\out\UBI-FIGHT"
            r"D:\Dataset\VAD\preprocessed\UCF"
        ],
        {
            "IITB": 0.8,
            "UBI-FIGHT": 0.8,
            "UCF": 0.8,
        },
        deepcopy(spath),
        counter
    )

    labeled_to_test(
        [
            r"D:\Dataset\VAD\preprocessed\IITB",
        ],
        deepcopy(spath),
        counter
    )
    return None


if __name__ == '__main__':
    main()
