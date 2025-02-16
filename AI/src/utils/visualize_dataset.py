from typing import *
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from AI.src.data.dataset import VideoFolderDataset
from AI.src.data.model import VideoMetadata

__all__ = ["prompt_dataset_statistics", "plot_dataset_statistics"]

def prompt_dataset_statistics(stats: Dict[str, Any]) -> str:
    """Tạo chuỗi mô tả thống kê dataset."""
    prompt = "=" * 50 + "\n"
    prompt += "DATASET STATISTICS\n"
    prompt += f"Total number of videos: {stats['total_samples']}\n"
    prompt += "Number of videos per class:\n"
    
    # Lặp qua từng lớp và số lượng video tương ứng
    for class_name, count in stats['class_count'].items():
        prompt += f"  - {class_name}: {count} videos\n"
    
    # Tính giá trị trung bình của FPS, độ phân giải, và thời gian video
    avg_fps = np.mean(list(stats['fps_count'].keys())) if stats['fps_count'] else 0
    avg_resolution = (
        int(np.mean([res[0] for res in stats['resolution'].keys()])),
        int(np.mean([res[1] for res in stats['resolution'].keys()]))
    ) if stats['resolution'] else (0, 0)
    min_duration = min(stats['duration'].keys()) if stats['duration'] else 0
    max_duration = max(stats['duration'].keys()) if stats['duration'] else 0
    avg_duration = np.mean(list(stats['duration'].keys())) if stats['duration'] else 0
    
    # Thêm thông tin vào chuỗi thống kê
    prompt += f"Average FPS: {avg_fps:.2f}\n"
    prompt += f"Average resolution: {avg_resolution}\n"
    prompt += f"Shortest duration: {min_duration:.2f} seconds\n"
    prompt += f"Longest duration: {max_duration:.2f} seconds\n"
    prompt += f"Average duration: {avg_duration:.2f} seconds\n"
    prompt += "=" * 50 + "\n"
    
    return prompt

def draw_bar_chart(x, y, title, xlabel, ylabel):
    """Vẽ biểu đồ cột với giá trị hiển thị rõ ràng."""
    plt.figure(figsize=(8, 6))  # Điều chỉnh kích thước biểu đồ
    ax = sns.barplot(x=x, y=y, palette="viridis", width=0.6)  # Sử dụng màu sắc hài hòa
    
    # Hiển thị số trên thanh, căn chỉnh để tránh đụng vào mép trên
    for i, value in enumerate(y):
        ax.text(i, value + max(y) * 0.02, str(value), ha="center", va="bottom", fontsize=8)
    
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, max(y) * 1.2)  # Mở rộng trục Y để số liệu không bị che
    plt.subplots_adjust(bottom=0.3)
    plt.show(block=False)

def plot_dataset_statistics(dataset: VideoFolderDataset):
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
    draw_bar_chart(list(sorted_class_names), list(sorted_class_counts), "Number of Videos per Class", "Class", "Number of Videos")
    
    # 2. So sánh số lượng video bất thường và bình thường
    anomaly_count = sum(count for cls, count in class_counts.items() if cls != "Normal")
    normal_count = class_counts.get("Normal", 0)
    
    draw_bar_chart(["Anomalies", "Normal"], [anomaly_count, normal_count], "Comparison of Anomalies vs Normal Videos", "Video Type", "Number of Videos")
    
    # 3. Phân phối độ dài video
    #
    bins = np.arange(0, 700, 100)  # Chia bins mỗi 100 giây 
    bin_counts, _ = np.histogram(durations, bins=bins)
    bin_labels = [f"[{bins[i]}, {bins[i+1]})" for i in range(len(bins)-1)]
    
    draw_bar_chart(bin_labels, bin_counts.tolist(), "Duration Distribution", "Seconds", "Number of Videos")

