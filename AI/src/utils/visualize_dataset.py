from typing import *
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from AI.src.data.dataset import VideoFolderDataset
from AI.src.data.model import VideoMetadata

__all__ = ["prompt_dataset_statistics", "plot_dataset_statistics"]

def prompt_dataset_statistics(stats: Dict[str, Any]) -> str:
    prompt = "=" * 50 + "\n"
    prompt += "DATASET STATISTICS\n"
    prompt += f"Total number of videos: {stats['total_samples']}\n"
    prompt += "Number of videos per class:\n"
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

    prompt += f"Average FPS: {avg_fps:.2f}\n"
    prompt += f"Average resolution: {avg_resolution}\n"
    prompt += f"Shortest duration: {min_duration:.2f} seconds\n"
    prompt += f"Longest duration: {max_duration:.2f} seconds\n"
    prompt += f"Average duration: {avg_duration:.2f} seconds\n"
    prompt += "=" * 50 + "\n"
    
    return prompt

def draw_bar_chart(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=x, y=y, palette="viridis")

    # Display count on each bar
    for i, value in enumerate(y):
        ax.text(i, value + 0.5, str(value), ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.show(block=False)

def plot_dataset_statistics(dataset: VideoFolderDataset):
    """
    Plot dataset statistics:
    - Bar chart (number of videos per class)
    - Histogram FPS
    - Histogram video duration
    """
    class_counts = defaultdict(int)
    fps_list = []
    durations = []  # List of video durations
    resolutions = []

    for stream_info, video_class in dataset:
        class_name = dataset.classes[video_class]  # Convert from index -> class name
        class_counts[class_name] += 1

        fps = getattr(stream_info, "frame_rate", 0)
        num_frames = getattr(stream_info, "num_frames", 0)
        resolution = (getattr(stream_info, "width", 0), getattr(stream_info, "height", 0))

        duration = num_frames / fps if fps > 0 else 0
        fps_list.append(fps)
        resolutions.append(resolution)
        durations.append(duration)

    # Calculate min, max, and average duration
    min_duration = np.min(durations) if durations else 0
    max_duration = np.max(durations) if durations else 0
    avg_duration = np.mean(durations) if durations else 0

    # 1. Bar chart - Number of videos per class (Ensure x is a list of class names)
    draw_bar_chart(list(class_counts.keys()), list(class_counts.values()), "Number of Videos per Class", "Class", "Number of Videos")

    # 2. Histogram - FPS Distribution
    anomaly_count = sum(count for cls, count in class_counts.items() if cls != "Normal")
    normal_count = class_counts.get("Normal", 0)

    labels = ["Anomalies", "Normal"]
    values = [anomaly_count, normal_count]

    plt.figure(figsize=(8, 5))
    ax = sns.barplot(x=labels, y=values, palette="coolwarm")

    # Add count on each bar
    for i, value in enumerate(values):
        ax.text(i, value + 0.5, str(value), ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.title("Comparison of Anomalies vs Normal Videos")
    plt.xlabel("Video Type")
    plt.ylabel("Number of Videos")
    plt.show(block=False)

    # 3. Histogram - Video Duration Distribution
    plt.figure(figsize=(10, 6))
    ax = sns.histplot(durations, bins='auto', kde=True, color="red")
    plt.title("Video Duration Distribution")
    plt.xlabel("Duration (seconds)")
    plt.ylabel("Number of Videos")

    # Add count on each bar
    for patch in ax.patches:
        if patch.get_height() > 0:
            ax.text(patch.get_x() + patch.get_width()/2, patch.get_height() + 0.5, 
                    str(int(patch.get_height())), ha="center", va="bottom", fontsize=10, fontweight="bold")
    # Limit x-axis if outliers exist
    plt.xlim(0, np.percentile(durations, 99))  # Trim the top 1% of values

    plt.show(block=False)


# def plot_dataset_statistics(dataset: VideoFolderDataset):
#     """
#     Vẽ các biểu đồ thống kê dataset:
#     - Biểu đồ cột (số lượng video theo class)
#     - Histogram FPS
#     - Histogram thời gian video
#     """
#     class_counts = defaultdict(int)
#     fps_list = []
#     durations = []  # Danh sách thời gian video
#     resolutions = []

#     for stream_info, video_class in dataset:
#         class_counts[video_class] += 1
#         fps = getattr(stream_info, "frame_rate", 0)
#         num_frames = getattr(stream_info, "num_frames", 0)
#         resolution = (getattr(stream_info, "width", 0), getattr(stream_info, "height", 0))

#         duration = num_frames / fps if fps > 0 else 0
#         fps_list.append(fps)
#         resolutions.append(resolution)
#         durations.append(duration)

#     # Tính thời gian ngắn nhất, dài nhất và trung bình
#     min_duration = np.min(durations) if durations else 0
#     max_duration = np.max(durations) if durations else 0
#     avg_duration = np.mean(durations) if durations else 0

#     # Đổi nhãn 0 -> "anomaly", 1 -> "normal"
#     class_labels = {0: "anomaly", 1: "normal"}
#     class_names = [class_labels.get(cls, str(cls)) for cls in class_counts.keys()]
#     class_values = list(class_counts.values())

#     # 1. Biểu đồ cột - Số lượng video theo class
#     plt.figure(figsize=(8, 5))
#     ax = sns.barplot(x=class_names, y=class_values, palette="coolwarm")
#     plt.title("Số lượng video theo lớp")
#     plt.xlabel("Lớp")
#     plt.ylabel("Số video")

#     # Thêm số lượng trên đầu cột
#     for i, value in enumerate(class_values):
#         ax.text(i, value + 1, str(value), ha="center", va="bottom", fontsize=12, fontweight="bold")

#     plt.show(block=False)

#     # 2. Histogram - Phân phối FPS
#     plt.figure(figsize=(10, 6))
#     ax = sns.histplot(fps_list, bins=10, kde=True, color="blue")
#     plt.title("Phân phối FPS")
#     plt.xlabel("FPS")
#     plt.ylabel("Số lượng video")

#     # Thêm số lượng trên đầu cột
#     for patch in ax.patches:
#         if patch.get_height() > 0:
#             ax.text(patch.get_x() + patch.get_width()/2, patch.get_height() + 0.5, 
#                     str(int(patch.get_height())), ha="center", va="bottom", fontsize=10, fontweight="bold")

#     plt.show(block=False)

#     # 3. Histogram - Phân phối thời gian video
#     plt.figure(figsize=(10, 6))
#     ax = sns.histplot(durations, bins=10, kde=True, color="red")
#     plt.title("Phân phối thời gian video")
#     plt.xlabel("Thời gian (giây)")
#     plt.ylabel("Số lượng video")

#     # Thêm số lượng trên đầu cột
#     for patch in ax.patches:
#         if patch.get_height() > 0:
#             ax.text(patch.get_x() + patch.get_width()/2, patch.get_height() + 0.5, 
#                     str(int(patch.get_height())), ha="center", va="bottom", fontsize=10, fontweight="bold")

#     plt.show(block=False)