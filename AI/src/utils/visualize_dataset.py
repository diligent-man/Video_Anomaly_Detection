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
    prompt += f"Tổng số video: {stats['total_samples']}\n"
    prompt += "Số lượng video theo lớp:\n"
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

    prompt += f"FPS trung bình: {avg_fps:.2f}\n"
    prompt += f"Độ phân giải trung bình: {avg_resolution}\n"
    prompt += f"Thời gian ngắn nhất: {min_duration:.2f} giây\n"
    prompt += f"Thời gian dài nhất: {max_duration:.2f} giây\n"
    prompt += f"Thời gian trung bình: {avg_duration:.2f} giây\n"
    prompt += "=" * 50 + "\n"
    
    return prompt



def draw_bar_chart(x, y, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    sns.barplot(x=x, y=y, palette="viridis")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45)
    plt.show(block=False)

def plot_dataset_statistics(dataset: VideoFolderDataset):
    """
    Vẽ các biểu đồ thống kê dataset:
    - Biểu đồ cột (số lượng video anomalies & normal)
    - Biểu đồ cột (số lượng video theo từng class)
    - Histogram FPS
    - Histogram thời gian video
    """
    class_counts = defaultdict(int)
    fps_list = []
    durations = []  

    for stream_info, target in dataset:
        video_class = dataset.classes[target]  # Lấy tên class thay vì số
        class_counts[video_class] += 1
        fps = getattr(stream_info, "frame_rate", 0)
        num_frames = getattr(stream_info, "num_frames", 0)

        duration = num_frames / fps if fps > 0 else 0
        fps_list.append(fps)
        durations.append(duration)

    # **Tách riêng anomalies & normal**
    num_anomalies = sum(count for cls, count in class_counts.items() if cls.lower() != "normal")
    num_normal = class_counts.get("Normal", 0)

    # **1️⃣ Biểu đồ cột - Số lượng video anomalies vs normal**
    plt.figure(figsize=(8, 5))
    sns.barplot(x=["Anomalies", "Normal"], y=[num_anomalies, num_normal], palette=["red", "blue"])
    plt.title("Số lượng video Anomalies vs Normal")
    plt.xlabel("Loại video")
    plt.ylabel("Số lượng video")
    plt.show(block=False)

    # **2️⃣ Biểu đồ cột - Số lượng video theo từng class**
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette="viridis")
    plt.xticks(rotation=45, ha="right")  # Xoay tên class dễ nhìn
    plt.title("Số lượng video theo lớp")
    plt.xlabel("Lớp")
    plt.ylabel("Số video")
    plt.show(block=False)

    # **3️⃣ Histogram - Phân phối FPS**
    plt.figure(figsize=(10, 6))
    sns.histplot(fps_list, bins=10, kde=True, color="blue")
    plt.title("Phân phối FPS")
    plt.xlabel("FPS")
    plt.ylabel("Số lượng video")
    plt.show(block=False)

    # **4️⃣ Histogram - Phân phối thời gian video**
    plt.figure(figsize=(10, 6))
    sns.histplot(durations, bins=10, kde=True, color="red")
    plt.title("Phân phối thời gian video")
    plt.xlabel("Thời gian (giây)")
    plt.ylabel("Số lượng video")
    plt.show(block=False)

    # **5️⃣ In thông tin thống kê**
    total_videos = num_anomalies + num_normal
    print("=" * 50)
    print(f"Tổng số video: {total_videos}")
    print(f"Anomalies: {num_anomalies} ({(num_anomalies / total_videos) * 100:.2f}%)")
    print(f"Normal: {num_normal} ({(num_normal / total_videos) * 100:.2f}%)")
    print("=" * 50)

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