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
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=x, y=y, palette="viridis")

    # Display count on each bar
    for i, value in enumerate(y):
        ax.text(i, value + 0.5, str(value), ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.subplots_adjust(bottom=0.3)
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

    plt.figure(figsize=(12, 6))
    ax = sns.barplot(x=labels, y=values, palette="coolwarm")

    # Add count on each bar
    for i, value in enumerate(values):
        ax.text(i, value + 0.5, str(value), ha="center", va="bottom", fontsize=12, fontweight="bold")

    plt.title("Comparison of Anomalies vs Normal Videos")
    plt.xlabel("Video Type")
    plt.ylabel("Number of Videos")
    plt.show(block=False)

    # 3. Histogram - Video Duration Distribution

# Giả sử durations chứa danh sách thời lượng video
    

# Xác định các khoảng thời lượng
    # bins = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 1000,1200,1800,2000,2500,3000,3500, np.inf]
    bins = [0, 60, 120, 180, 240, 300, 360, 420, 480, 540,  np.inf]
    bin_labels = [f"[{int(bins[i])}, {int(bins[i+1])}[" if bins[i+1] != np.inf else f"[{int(bins[i])}, ∞[" for i in range(len(bins)-1)]


# Phân loại số lượng video trong từng khoảng
    bin_counts = np.histogram(durations, bins=bins)[0]

# Vẽ biểu đồ cột
    plt.figure(figsize=(12, 6))
    plt.bar(bin_labels, bin_counts, color='blue')

# Thêm số lượng video lên từng cột
    for i, count in enumerate(bin_counts):
        plt.text(i, count + 2, str(count), ha='center', fontsize=12)

    plt.xlabel("Seconds")
    plt.ylabel("Number of videos")
    plt.title("Videos Durations Distribution")
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    plt.subplots_adjust(bottom=0.2)

    plt.show()



