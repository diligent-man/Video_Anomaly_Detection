import os
import requests
import numpy as np
from pathlib import Path

from .video_utils import get_video_info, plot_vad_animation

# Định nghĩa thư mục lưu trữ
TMP_DIR = os.getenv("TMP_DIR", Path("tmp").resolve())
VIDEO_DIR = f"{TMP_DIR}{os.sep}videos"
SCORES_DIR = f"{TMP_DIR}{os.sep}scores"
PLOTS_DIR = f"{TMP_DIR}{os.sep}plots"

# Tạo các thư mục nếu chưa tồn tại
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(SCORES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# URL for the inference API
VAD_API_URL: str = os.getenv("VAD_ENDPOINT", "http://0.0.0.0:6968")


def run_vad_model(video_path):
    """ 
    Gửi video đến API và lưu anomaly scores trả về 
    """
    try:
        # Chuẩn bị file để gửi
        with open(video_path, 'rb') as video_file:
            video_filename = os.path.basename(video_path)
            files = {'file': (video_filename, video_file, 'video/mp4')}

            # Gửi request đến API
            response = requests.post(f"{VAD_API_URL}/infer", files=files)
            
            # Kiểm tra response
            if response.status_code != 200:
                raise Exception(f"API request failed with status code {response.status_code}: {response.text}")
            
            # Lấy dữ liệu anomaly scores từ response
            response_data = response.json()
            if 'preds' not in response_data:
                raise Exception("No predictions in API response")
            
            scores = response_data['preds']
            fps = response_data.get('fps', 30)  
            # Lưu scores vào file
            scores_file = get_scores_path(video_filename)
            np.save(scores_file, np.array(scores))
            return str(scores_file), fps
            
    except Exception as e:
        print(f"Error in run_vad_model: {str(e)}")
        raise


def get_scores_path(video_name: str) -> Path:
    """Generate the expected scores file path for a video"""
    return Path("{SCORES_DIR}{os.sep}{Path(video_name).stem}_scores.npy")


def get_plot_path(video_name: str) -> Path:
    """Generate the expected plot animation file path for a video"""
    return Path(f"{PLOTS_DIR}{os.sep}{Path(video_name).stem}_plot.mp4")


def generate_plot(video_name, scores=None, fps=None):
    """Generate plot animation for a video based on its anomaly scores"""
    try:
        plot_path = get_plot_path(video_name)
        
        # If scores not provided, load them
        if scores is None:
            scores_file = get_scores_path(video_name)
            if not scores_file.exists():
                return None
            scores = np.load(scores_file).tolist()
        
        if fps is None:
            video_path = f"{VIDEO_DIR}{os.sep}{video_name}"
            fps, _, _ = get_video_info(video_path)
        
        # Generate plot animation
        plot_path_str = plot_vad_animation(
            scores, 
            fps=fps, 
            save_path=str(plot_path)
        )
        return plot_path_str
    except Exception as e:
        print(f"Error generating plot: {str(e)}")
        return None
