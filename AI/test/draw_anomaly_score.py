import os
import torch
import numpy as np
import pandas as pd
from typing import Tuple, List
from matplotlib.ticker import MaxNLocator
from glob import glob
from matplotlib import pyplot as plt
from AI.src.data.dataset.VADFrameLevelDataset import VADFrameLevelDataset
plt.switch_backend("tkagg")
import numpy as np
import matplotlib.pyplot as plt
from typing import List


def draw_anomaly_graph(
    preds: List[List[float]], 
    labels: List[int], 
    legends: List[str], 
    name: str
):
    T = len(labels)
    x = np.arange(T)

    fig, ax = plt.subplots(figsize=(10, 4))

    for pred, legend in zip(preds, legends):
        ax.plot(x, pred, label=legend, linewidth=1.5)

    labels_np = np.array(labels)
    anomaly_regions = np.where(labels_np == 1)[0]

    if len(anomaly_regions) > 0:
        start = anomaly_regions[0]
        for i in range(1, len(anomaly_regions)):
            if anomaly_regions[i] != anomaly_regions[i - 1] + 1:
                ax.axvspan(start, anomaly_regions[i - 1], color='red', alpha=0.3)
                start = anomaly_regions[i]
        ax.axvspan(start, anomaly_regions[-1], color='red', alpha=0.3)

    ax.set_yticks(np.linspace(0, 1, 11)) 
    ax.set_xticks(np.linspace(0, T, 6))   
    ax.set_xlabel("Frame")
    ax.set_ylabel("Score")
    
    ax.spines['left'].set_position('zero')  
    ax.spines['bottom'].set_position('zero')  
    ax.spines['right'].set_color('black')  
    ax.spines['top'].set_color('black') 
    
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')  

    ax.set_xlim(left=0) 
    ax.set_ylim(bottom=0, top=1.05) 

    ax.legend(loc='upper left', bbox_to_anchor=(1, 1), title="Legend")

    ax.set_title(name)
    plt.tight_layout()
    plt.show()


def main() -> None:
    label_df = pd.read_csv("D:/final/test/label.csv", header=None)

    video_names = []
    for _, row in label_df.iterrows():
        video_path = os.path.join("D:/final/test", row[0])
        if os.path.exists(video_path):
            name = os.path.basename(video_path).replace(".pt", "")
            video_names.append(name)

    ds = VADFrameLevelDataset(
        "D:/final/test",
        "D:/final/test/label.csv",
        "v4"
    )

    for idx, (video, labels) in enumerate(ds):
        name = video_names[idx]

        T = video.shape[0]
        from scipy.ndimage import gaussian_filter1d
        clip_pred = gaussian_filter1d(np.random.rand(T), sigma=10)
        s3d_pred = gaussian_filter1d(np.random.rand(T), sigma=8)
        i3d_pred = gaussian_filter1d(np.random.rand(T), sigma=6)


if __name__ == '__main__':
    main()
