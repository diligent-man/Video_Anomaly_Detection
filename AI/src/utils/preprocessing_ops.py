import os
import re
import cv2
import shutil
import pandas as pd
import ffmpeg
from docx import Document

__all__ = [ "remove_videos_from_csv", "find_long_videos", "delete_long_videos", "delete_videos_from_folders"]

def remove_videos_from_csv(txt_file, csv_file):
    video_names = set()
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" - ")
            if parts:
                video_name = parts[0].replace(".mp4", "").strip()
                video_names.add(video_name)

    df = pd.read_csv(csv_file, header=None, dtype=str).astype(str)
    df_filenames = df.iloc[:, 0].apply(lambda x: x.split("/")[-1].replace(".mp4", "").strip())
    mask = df_filenames.isin(video_names)
    removed_rows = df[mask]

    if not removed_rows.empty:
        backup_csv = csv_file + ".bak"
        if not os.path.exists(backup_csv):
            df.to_csv(backup_csv, index=False, header=False)
        df_filtered = df[~mask]
        df_filtered.to_csv(csv_file, index=False, header=False)
        print(f"\n Hoàn thành! File {csv_file} sau khi xóa có {len(df_filtered)} dòng.")
    else:
        print("\n Không có dòng nào bị xóa.")

def find_long_videos(folder_path, min_duration=15):
    def get_video_duration(video_path):
        try:
            probe = ffmpeg.probe(video_path)
            return float(probe['format']['duration']) / 60
        except Exception as e:
            print(f"Lỗi khi lấy thời lượng video {video_path}: {e}")
            return None

    long_videos = []
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith(('.mp4', '.mkv', '.avi', '.mov', '.wmv')):
                video_path = os.path.join(root, file)
                duration = get_video_duration(video_path)
                if duration and duration > min_duration:
                    long_videos.append((file, duration))
    return long_videos

def delete_long_videos(folder_path, max_duration=15):
    if not os.path.exists(folder_path):
        print("Thư mục không tồn tại!")
        return
    
    for root, _, files in os.walk(folder_path):
        for file_name in files:
            file_path = os.path.join(root, file_name)
            if file_name.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
                cap = cv2.VideoCapture(file_path)
                if not cap.isOpened():
                    print(f"Không thể mở video: {file_path}")
                    continue
                
                fps = cap.get(cv2.CAP_PROP_FPS)
                frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                duration = frame_count / fps
                cap.release()
                
                if duration > max_duration * 60:
                    os.remove(file_path)
                    print(f"Đã xóa: {file_path} ({duration / 60:.2f} phút)")

def delete_videos_from_folders(file_txt, root_folder):
    video_names = set()
    with open(file_txt, "r", encoding="utf-8") as f:
        for line in f:
            match = re.match(r"^(\S+\.mp4)", line.strip())
            if match:
                video_names.add(match.group(1))
    
    for subfolder in os.listdir(root_folder):
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):
            for file in os.listdir(subfolder_path):
                if file in video_names:
                    file_path = os.path.join(subfolder_path, file)
                    try:
                        os.remove(file_path)
                        print(f"Đã xóa: {file_path}")
                    except Exception as e:
                        print(f"Lỗi khi xóa {file_path}: {e}")
