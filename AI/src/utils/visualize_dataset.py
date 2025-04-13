import os

from typing import Dict, Any
from collections import defaultdict


import numpy as np
import seaborn as sns

from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator


from .ANSIColor import ANSIColor
from ..data.dataset import VideoFolderDataset


__all__ = [
    "prompt_dataset_statistics",
    "plot_dataset_statistics"
]


def prompt_dataset_statistics(stats: Dict[str, Any]) -> str:
    """
    Generates a formatted string displaying dataset statistics.

    Args:
        stats (Dict[str, Any]): A dictionary containing dataset statistics.

    Returns:
        str: A formatted string with dataset statistics.
    """
    prompt = f"""{'-' * 40}  {ANSIColor().CYAN}DATASET STATISTICS{ANSIColor().RESET}  {'-' * 40}
DATASET STATISTICS
Total number of videos: {stats['total_samples']}
Number of videos per class:
"""

    for class_name, count in stats['class_count'].items():
        prompt += f"  - {class_name}: {count} videos\n"
    
    # Calculate average FPS, resolution, and duration
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
    """
    Plots various dataset statistics, including class distribution, anomaly comparison, and duration distribution.

    Args:
        dataset (VideoFolderDataset): The dataset containing video samples.
        **kwargs: Additional parameters for customization.
    """
    class_counts = defaultdict(int)  # Store the number of videos per class
    fps_list = []  # Store the FPS values
    durations = []  # Store video durations
    resolutions = []  # Store video resolutions

    # Iterate through the dataset to collect statistics
    for stream_info, video_class in dataset:
        class_name = dataset.classes[video_class]  # Get the class name
        class_counts[class_name] += 1

        # Extract video properties
        fps = getattr(stream_info, "frame_rate", 0)
        num_frames = getattr(stream_info, "num_frames", 0)
        resolution = (getattr(stream_info, "width", 0), getattr(stream_info, "height", 0))

        # Compute video duration (avoid division by zero)
        duration = num_frames / fps if fps > 0 else 0
        fps_list.append(fps)
        resolutions.append(resolution)
        durations.append(duration)

    # Sort class counts in ascending order
    sorted_classes = sorted(class_counts.items(), key=lambda item: item[1])
    sorted_class_names, sorted_class_counts = zip(*sorted_classes)

    # 1. Plot the number of videos per class
    # Filter out classes containing "Normal"
    filtered_classes = [(name, count) for name, count in zip(sorted_class_names, sorted_class_counts) if "Normal" not in name]

    # Extract filtered class names and counts
    filtered_class_names, filtered_class_counts = zip(*filtered_classes) if filtered_classes else ([], [])

    # Plot the filtered data
    draw_bar_chart(list(filtered_class_names),
                   list(filtered_class_counts),
                   "Videos per Class",
                   "",
                   "Number of Videos",
                   {"rotation": 45, "ha": "right"},
                   None,
                   None,
                   {"bottom": 0, "top": max(filtered_class_counts) * 1.2} if filtered_class_counts else
                   {"bottom": 0, "top": 1},
                   **kwargs
               )

    # 2. Compare the number of anomaly and normal videos
    anomaly_count = sum(count for cls, count in class_counts.items() if cls != "Normal")
    normal_count = class_counts.get("Normal", 0)

    draw_bar_chart(["Anomalies", "Normal"], [anomaly_count, normal_count],
                   "Anomaly vs Normal videos",
                   None,
                   None,
                   {"rotation": 0, "ha": "center"},
                   None,
                   None,
                   {"bottom": 0, "top": max([anomaly_count, normal_count]) * 1.2},
                   True,
                   width=0.5,  # Chỉ áp dụng cho biểu đồ này
                   **kwargs
                   )
    # quay anomaly với normal ngang lại , bóp cột 2/3 hay 1/2 nhỏ lại

    # 3. Plot video duration distribution
    bins = np.concatenate([np.arange(0, 120, 30), np.arange(120, 900, 120),np.arange(900, 960, 60), [np.inf]])
    # bins = np.concatenate([np.arange(0, 120, 30), np.arange(120, 600, 120),np.arange(600, 660, 60), [np.inf]])

    bin_counts, _ = np.histogram(durations, bins=bins)
    bin_labels = [f"{int(bins[i])}-{int(bins[i+1])}" for i in range(len(bins)-2)] + ["900+"]



    draw_bar_chart(bin_labels, bin_counts.tolist(),
                   "Time Distribution",
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
                   fpath: str = None,
                   width: float = 0.8  # Thêm tham số width với giá trị mặc định
                   ) -> None:
    """
    Draws a bar chart using Seaborn and Matplotlib.

    Args:
        x (Any): X-axis labels.
        y (Any): Y-axis values.
        title (str): Title of the chart.
        xlabel (str, optional): Label for the X-axis. Defaults to None.
        ylabel (str, optional): Label for the Y-axis. Defaults to None.
        xticks (dict, optional): X-axis tick properties. Defaults to None.
        yticks (dict, optional): Y-axis tick properties. Defaults to None.
        xlim (dict, optional): X-axis limits. Defaults to None.
        ylim (dict, optional): Y-axis limits. Defaults to None.
        force_y_int (bool, optional): If True, forces integer values on the Y-axis. Defaults to False.
        fpath (str, optional): File path to save the plot. Defaults to None.
    """
    if xticks is None:
        xticks = {}

    if yticks is None:
        yticks = {}

    if xlim is None:
        xlim = {}

    if ylim is None:
        ylim = {}

    plt.figure(figsize=(8, 6))

    # Giảm độ sáng màu đen bằng cách dùng màu xám nhạt hơn
    if title == "Anomaly vs Normal videos":
        colors = ["red", "blue"]  # Đỏ cho bất thường, xanh cho bình thường
    else:
        colors = ["#c2c2c2"] * len(y)  # Xám thay vì đen (#808080 là màu xám trung bình)

    ax = sns.barplot(x=x, y=y, palette=colors, legend=False, width=width)

    if force_y_int:
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Display values on top of each bar
    for i, value in enumerate(y):
        ax.text(i, value + max(y) * 0.02, str(value), ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    plt.xticks(**xticks)

    # Đặt màu đỏ cho nhãn "0-120" và "900+"
    xticklabels = ax.get_xticklabels()
    for label in xticklabels:
        if label.get_text() in ["0-30", "30-60", "60-90", "90-120"]:
            label.set_color("green")
        elif label.get_text() == "900+":
            label.set_color("red")

    plt.yticks([])  # Remove Y-axis labels as required

    plt.xlim(**xlim)
    plt.ylim(**ylim)  # Expand Y-axis limits to avoid label overlap
    plt.subplots_adjust(bottom=0.3)

    if fpath is not None and os.path.isdir(fpath):
        plt.savefig(f"{fpath}{os.sep}{title}")
    else:
        plt.show()
