import os.path
from typing import Dict, Any
from collections import defaultdict
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator

from .ANSIColor import ANSIColor
from ..data.dataset import VideoFolderDataset


__all__ = ["prompt_dataset_statistics", "plot_dataset_statistics"]


def prompt_dataset_statistics(stats: Dict[str, Any]) -> str:
    prompt = f"""{'-' * 40}  {ANSIColor().CYAN}DATASET STATISTICS{ANSIColor().RESET}  {'-' * 40}
DATASET STATISTICS
Total number of videos: {stats['total_samples']}
Number of videos per class:
"""

    for class_name, count in stats['class_count'].items():
        prompt += f"  - {class_name}: {count} videos\n"
    
    avg_fps = np.mean(list(stats['fps_count'].keys())) if stats['fps_count'] else 0
    avg_resolution = (
        int(np.mean([res[0] for res in stats['resolution'].keys()])),
        int(np.mean([res[1] for res in stats['resolution'].keys()]))
    ) if stats['resolution'] else (0, 0)
    min_duration = min(stats['duration'].keys()) if stats['duration'] else 0
    max_duration = max(stats['duration'].keys()) if stats['duration'] else 0
    avg_duration = np.mean(list(stats['duration'].keys())) if stats['duration'] else 0

    prompt += f"""Average FPS: {avg_fps:.2f}
Average resolution: {avg_resolution}
Shortest duration: {min_duration:.2f} seconds
Longest duration: {max_duration:.2f} seconds
Average duration: {avg_duration:.2f} seconds
{'-' * (84 + len("DATASET STATISTICS"))}
"""
    return prompt


def plot_dataset_statistics(dataset: VideoFolderDataset, **kwargs) -> None:
    """Vẽ các biểu đồ thống kê từ dataset."""
    class_counts = defaultdict(int)  # Số lượng video theo từng lớp
    fps_list = []  # Danh sách FPS của video
    durations = []  # Danh sách thời gian video
    resolutions = []  # Danh sách độ phân giải video

    # Duyệt qua dataset và thu thập thông tin
    for stream_info, video_class in dataset:
        class_name = dataset.classes[video_class]  # Lấy tên lớp video
        class_counts[class_name] += 1

        # Lấy thông tin về FPS, số frame, độ phân giải
        fps = getattr(stream_info, "frame_rate", 0)
        num_frames = getattr(stream_info, "num_frames", 0)
        resolution = (getattr(stream_info, "width", 0), getattr(stream_info, "height", 0))

        # Tính thời gian video (tránh chia cho 0)
        duration = num_frames / fps if fps > 0 else 0
        fps_list.append(fps)
        resolutions.append(resolution)
        durations.append(duration)

    # Sắp xếp số lượng video theo lớp
    sorted_classes = sorted(class_counts.items(), key=lambda item: item[1])
    sorted_class_names, sorted_class_counts = zip(*sorted_classes)

    # 1. Vẽ biểu đồ số video theo lớp
    # Lọc bỏ các lớp có chứa "Normal"
    filtered_classes = [(name, count) for name, count in zip(sorted_class_names, sorted_class_counts) if "Normal" not in name]

    # Tách danh sách sau khi lọc
    filtered_class_names, filtered_class_counts = zip(*filtered_classes) if filtered_classes else ([], [])

    # Vẽ biểu đồ với dữ liệu đã lọc
    draw_bar_chart(list(filtered_class_names),
               list(filtered_class_counts),
               "Number of Videos per Class",
               "Class", "Number of Videos",
               {"rotation": 45, "ha": "right"}, None,
               None, {"bottom": 0, "top": max(filtered_class_counts) * 1.2} if filtered_class_counts else {"bottom": 0, "top": 1},
               **kwargs
               )


    # 2. So sánh số lượng video bất thường và bình thường
    anomaly_count = sum(count for cls, count in class_counts.items() if cls != "Normal")
    normal_count = class_counts.get("Normal", 0)

    draw_bar_chart(["Anomalies", "Normal"], [anomaly_count, normal_count],
                   "Anomaly vs Normal videos",
                   None, None,
                   {"rotation": 45, "ha": "right"}, None,
                   None,  {"bottom": 0, "top": max([anomaly_count, normal_count]) * 1.2},
                   True,
                   **kwargs
                   )

    # 3. Phân phối độ dài video
    bins = np.arange(0, 700, 100)  # Chia bins mỗi 100 giây
    bin_counts, _ = np.histogram(durations, bins=bins)
    bin_labels = [f"[{bins[i]}, {bins[i + 1]})" for i in range(len(bins) - 1)]

    draw_bar_chart(bin_labels, bin_counts.tolist(),
                   "Duration Distribution",
                   "Seconds", "Number of Videos",
                   {"rotation": 45, "ha": "right"}, None,
                   None, {"bottom": 0, "top": max(bin_counts.tolist()) * 1.2},
                   **kwargs
                   )
########################################################################################################################


def draw_bar_chart(x: Any, y: Any, title: str,
                   xlabel: str = None, ylabel: str = None,
                   xticks: dict = None, yticks: dict = None,
                   xlim: dict = None, ylim: dict = None,
                   force_y_int: bool = False,
                   fpath: str = None
                   ) -> None:
    if xticks is None:
        xticks = {}

    if yticks is None:
        yticks = {}

    if xlim is None:
        xlim = {}

    if ylim is None:
        ylim = {}

    plt.figure(figsize=(8, 6))
    ax = sns.barplot(x=x, y=y, hue=y, palette="viridis", legend=False)

    if force_y_int:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Hiển thị số trên thanh, căn chỉnh để tránh đụng vào mép trên
    for i, value in enumerate(y):
        ax.text(i, value + max(y) * 0.02, str(value), ha="center", va="bottom", fontsize=8)

    plt.title(title)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(**xticks)
    plt.yticks(**yticks)

    plt.xlim(**xlim)
    plt.ylim(**ylim)  # Mở rộng trục Y để số liệu không bị che
    plt.subplots_adjust(bottom=0.3)

    if fpath is not None and os.path.isdir(fpath):
        plt.savefig(f"{fpath}{os.sep}{title}")
    else:
        plt.show()
