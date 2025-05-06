import os
import numpy as np
from functools

from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from Web.src.be.src.utils.video_utils import find_anomaly_regions


os.environ["WINDOW_LENGTH"] = str(15)
os.environ["POLYORDER"] = str(6)

os.environ["HEIGHT"] = str(0.9)
os.environ["THRESHOLD"] = str(None)
os.environ["DISTANCE"] = str(None)
os.environ["PROMINENCE"] = str(0.00001)
os.environ["WIDTH"] = str(3)
os.environ["WLEN"] = str(None)
os.environ["REL_HEIGHT"] = str(0.8)
os.environ["PLATEAU_SIZE"] = str(None)

os.environ["MERGE_GAP"] = str(5)


# Hàm cập nhật cho animation
def update(frame: int):
    print(frame)
    line.set_data(x_values[:frame + 1], y_values[:frame + 1])
    point.set_data([x_values[frame]], [y_values[frame]])
    return line, point


def plot_vad_animation(anomaly_scores: np.ndarray, fps: int = 30, save_path: str = "vad_plot.mp4"):
    """
    Creates an animated plot of video anomaly detection scores.

    Parameters:
    -----------
    anomaly_scores : list or array
        The anomaly scores to visualize
    fps : int
        Frames per second for the animation
    save_path : str
        Path to save the animation file
    high_threshold : float or None
        Threshold for peak height detection (None = use default)
    low_threshold : float or None
        Threshold for peak prominence (None = use default)
    window_length : int or None
        Length of the smoothing window (None = use default)
    polyorder : int or None
        Polynomial order (None = use default)

    Returns:
    --------
    str
        Path to the saved animation file
    """
    total_frames: int = len(anomaly_scores)
    anomaly_regions, processed_scores, peaks = find_anomaly_regions(anomaly_scores)

    # Tạo figure
    fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)

    # Vẽ đường đồ thị cho anomaly scores
    line, = ax.plot([], [], lw=2, label="Anomaly Score", color='blue')
    point, = ax.plot([], [], 'ro', ms=6)  # Current frame marker

    ax.set_xlabel("Frames")
    ax.set_ylabel("Anomaly Score")
    ax.set_xlim(0, total_frames)
    ax.set_ylim(0, max(1.0, np.max(anomaly_scores) * 1.1))
    ax.legend()

    # Highlight các vùng anomaly
    for start, end in anomaly_regions:
        ax.axvspan(start, end, color='red', alpha=0.3)

    # Vẽ peaks nếu có
    if len(peaks) > 0:
        ax.plot(peaks, processed_scores[peaks], "x", color='darkred', ms=8)

    # Dữ liệu x và y cho animation
    x_values = np.arange(total_frames)
    y_values = processed_scores

    ani: FuncAnimation = FuncAnimation(fig, update, frames=range(0, total_frames, 1), interval=1000/fps, blit=True)
    writer: FFMpegWriter = FFMpegWriter(fps=min(30, int(fps)), metadata=dict(artist='Video Anomaly Detection'), bitrate=800)
    ani.save(save_path, writer=writer)
    plt.close(fig)
    return save_path


def main() -> None:
    scores = np.load("/home/trong/Downloads/Arrest001_x264_scores.npy").tolist()
    plot_vad_animation(scores, save_path="/home/trong/Downloads/Arrest001_x264_scores.mp4")
    return None

    
if __name__ == '__main__':
    main()