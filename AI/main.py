import torch
from matplotlib import pyplot as plt

from AI.src.data.dataset import VADFrameLevelDataset

plt.switch_backend("tkagg")


def main() -> None:
    # Assume frame is labeled from 0
    ds = VADFrameLevelDataset(
        "/home/trong/Downloads/Dataset/VAD/final/test",
        "label.csv",
        "v4",
    )

    for video, labels in ds:
        print(video.shape)
        print(labels.tolist())
        draw_anamaly_graph(
            preds: List[List[float]],
            labels: List[List[int]],
            legends: List[str],
            name: str
        )

    # x = torch.arange(1, Total_frames, Total_frames)
    # scores = Frames_Score
    # scores1 = scores.reshape((scores.shape[1],))
    # scores1 = savitzky_golay(scores1, 101, 3)
    # plt.close()

    return None


if __name__ == '__main__':
    main()