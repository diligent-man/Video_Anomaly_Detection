import os
import re

from pathlib import Path
from typing import Tuple, List

import ffmpeg
import pandas as pd

from ..constant import VIDEO_EXTENSIONS

__all__ = [
    "modify_fpath",
    "find_long_videos",
    "update_data_from_delete",
    "update_label_from_delete"
]


def modify_fpath(file_path, output_path) -> None:
    """
    Modify file paths in file_path and save them to output_path.

    Usage:
    modify_csv_paths("input.csv", "output.csv")
    """
    def modify_path(row: Tuple[str, str, int, int, int, int]) -> str:
        """
        :param row: tuple includes (fpath, normal|anomaly, start1, end1, start2, end2)
        """
        path, label = row[0], row[1]
        filename: str = Path(path).name
        path = os.path.join("anomaly", label.lower(), filename) if label.lower() == "anomaly" else \
            os.path.join("normal", filename)
        return path

    df: pd.DataFrame = pd.read_csv(file_path, header=None)
    df[0] = df.apply(modify_path, axis=1)
    df.drop(columns=1, axis="columns", inplace=True)
    df.to_csv(output_path, index=False, header=False)
    print(f"\n Completed! File saved to {output_path}.")


def find_long_videos(folder_path, min_duration=900) -> List[Tuple[str, float]]:
    """
    Find videos longer than min_duration seconds in the specified folder.

    Usage:
    long_videos = find_long_videos("./videos", 900)
    """

    def get_video_duration(video_path):
        try:
            probe = ffmpeg.probe(video_path)  # Use ffmpeg to get video information
            return float(probe['format']['duration'])  # Get video duration
        except Exception as e:
            print(f"Error retrieving duration for video {video_path}: {e}")
            return None

    results: List[Tuple[str, float]] = []
    for root, _, files in os.walk(folder_path):  # Traverse all files in the directory
        results = [
            f for f in files if (
                f.lower().endswith(VIDEO_EXTENSIONS) and
                get_video_duration(os.path.join(root, f)) > min_duration
            )
        ]
    return results


def update_data_from_delete(file_txt, root_folder) -> None:
    """
    Delete videos listed in file_txt from the root_folder.

    Usage:
    update_data_from_delete("deleted_videos.txt", "./video_folder")
    """
    video_names = set()
    with open(file_txt, "r", encoding="utf-8") as f:
        for line in f:
            # Extract video from the list
            match = re.match(r"^(\S+\.[{}])".format("|".join(VIDEO_EXTENSIONS)), line.strip())
            if match:
                video_names.add(match.group(1))

    for subfolder in os.listdir(root_folder):  # Traverse all subdirectories
        subfolder_path = os.path.join(root_folder, subfolder)
        if os.path.isdir(subfolder_path):  # Check if it is a directory
            for file in os.listdir(subfolder_path):
                if file in video_names:  # If file is in the deletion list
                    file_path = os.path.join(subfolder_path, file)
                    try:
                        os.remove(file_path)  # Delete file
                        print(f"Deleted: {file_path}")
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")


def update_label_from_delete(txt_file, csv_file) -> None:
    """
    Remove rows from csv_file if the video name appears in txt_file.

    Usage:
    update_label_from_delete("deleted_videos.txt", "labels.csv")
    """
    video_names = set()
    with open(txt_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(" - ")
            if parts:
                video_name = parts[0].replace(".mp4", "").strip()
                video_names.add(video_name)  # Store video names in a set
    print(f"\n {len(video_names)} videos to be removed.")

    df = pd.read_csv(csv_file, header=None, dtype=str).astype(str)  # Read CSV file
    df_filenames = df.iloc[:, 0].apply(lambda x: x.split("/")[-1].replace(".mp4", "").strip())  # Extract filenames
    mask = df_filenames.isin(video_names)  # Mark rows to delete
    removed_rows = df[mask]  # Get deleted rows

    if not removed_rows.empty:
        df_filtered = df[~mask]  # Keep only non-deleted rows
        df_filtered.to_csv(csv_file, index=False, header=False)  # Save updated CSV file
        print(f"\n Completed! {csv_file} now has {len(df_filtered)} rows.")
    else:
        print("\n No rows were deleted.")
