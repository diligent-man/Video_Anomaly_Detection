import torch
from pprint import pprint as pp


def main() -> None:
    ckpt = torch.load("/home/trong/Downloads/Local/Source/Python/semester_9/AIP391/Video_anomaly_detection/AI/results/AIP391/single/train/legacy/input_2d/Mlflow/528579867362157536/22d98f9da1ec478c8ebff673474dc70c/artifacts/ckpt/best_epoch13_step2937.pt")
    pp(ckpt["config"]["Global"])
    pp(ckpt["config"]["Optim"])
    return None


if __name__ == '__main__':
    main()