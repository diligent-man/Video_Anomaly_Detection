import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

from .constant import smooth_signal, PeakDetector

__all__ = ["plot_vad_animation", "find_anomaly_regions", "get_video_info"]

# Default parameters that can be overridden by environment variables
MERGE_GAP = int(os.environ.get("MERGE_GAP", 7))


def get_video_info(video_path):
    """
    Get video information using ffmpeg-python library
    Returns: fps, total_frames, video_duration
    """
    import ffmpeg
    
    try:
        # Get video information using ffprobe
        probe = ffmpeg.probe(str(video_path))
        
        # Extract video stream information
        video_stream = next((stream for stream in probe['streams'] 
                           if stream['codec_type'] == 'video'), None)
        
        if video_stream is None:
            raise ValueError("Không tìm thấy video stream!")
        
        # Get frame rate
        if 'r_frame_rate' in video_stream:
            # Frame rate often comes as a fraction (e.g., "24/1")
            framerate_str = video_stream['r_frame_rate']
            if '/' in framerate_str:
                num, den = map(int, framerate_str.split('/'))
                fps = num / den if den else 0
            else:
                fps = float(framerate_str)
        else:
            # Fallback to avg_frame_rate if r_frame_rate is not available
            framerate_str = video_stream.get('avg_frame_rate', '0/1')
            if '/' in framerate_str:
                num, den = map(int, framerate_str.split('/'))
                fps = num / den if den else 0
            else:
                fps = float(framerate_str)
        
        # Get total frames - directly from stream if available
        if 'nb_frames' in video_stream and video_stream['nb_frames'].isdigit():
            total_frames = int(video_stream['nb_frames'])
        else:
            # Calculate from duration if frame count is not provided
            if 'duration' in video_stream:
                duration = float(video_stream['duration'])
            else:
                # If stream doesn't have duration, check format
                duration = float(probe['format'].get('duration', 0))
            
            # Calculate frames from duration and fps
            total_frames = int(duration * fps)
        
        # Calculate video duration
        video_duration = total_frames / fps if fps > 0 else 0
        
        return fps, total_frames, video_duration
        
    except ffmpeg.Error as e:
        # FFmpeg error messages are usually in stderr
        error_message = e.stderr.decode('utf-8') if hasattr(e, 'stderr') else str(e)
        raise ValueError(f"FFmpeg error: {error_message}")
    except Exception as e:
        raise ValueError(f"Lỗi khi đọc thông tin video: {str(e)}")


def find_anomaly_regions(anomaly_scores, high_threshold=None, low_threshold=None, 
                         window_length=None, polyorder=None):
    """
    Find anomaly regions using peak detection with signal filtering.

    Parameters:
    -----------
    anomaly_scores : list or array
        The anomaly scores to analyze
    high_threshold : float or None
        Threshold for peak height detection (None = use default)
    low_threshold : float or None
        Threshold for peak prominence (None = use default)
    window_length : int or None
        Length of the smoothing window (None = use default)
    polyorder : int or None
        Polynomial order for filter (None = use default)

    Returns:
    --------
    tuple
        (anomaly_regions, processed_scores, peaks)
        - anomaly_regions: list of (start, end) tuples
        - processed_scores: smoothed signal
        - peaks: array of detected peak indices
    """
    # Create configuration objects with provided or default values
    smoother = smooth_signal()
    if window_length is not None:
        smoother.window_length = window_length
    if polyorder is not None:
        smoother.polyorder = polyorder
        
    peak_detector = PeakDetector()
    if high_threshold is not None:
        peak_detector.height = high_threshold
    if low_threshold is not None:
        peak_detector.prominence = low_threshold
    
    # Convert to numpy array if not already
    anomaly_scores = np.array(anomaly_scores)
    total_frames = len(anomaly_scores)

    # Apply smoothing using the smoother dataclass
    processed_scores = smoother.apply(anomaly_scores)

    # Find peaks using the peak_detector dataclass
    peaks, properties = peak_detector.detect(processed_scores)

    # Handle the case with no detected peaks
    if len(peaks) == 0:
        return [], processed_scores, peaks

    # Calculate peak widths for determining anomaly regions
    widths, width_heights, left_ips, right_ips = peak_detector.get_peak_regions(processed_scores, peaks)

    # Create anomaly regions based on peak widths
    anomaly_regions = []
    for i, peak in enumerate(peaks):
        start = max(0, int(left_ips[i]))
        end = min(total_frames-1, int(right_ips[i]))

        # Extend region to include nearby high values
        while start > 0 and processed_scores[start-1] > peak_detector.prominence:
            start -= 1
        while end < total_frames-1 and processed_scores[end+1] > peak_detector.prominence:
            end += 1

        anomaly_regions.append((start, end))

    # Merge overlapping regions
    if anomaly_regions:
        anomaly_regions.sort(key=lambda x: x[0])
        merged_regions = [anomaly_regions[0]]

        for current in anomaly_regions[1:]:
            prev = merged_regions[-1]
            if current[0] <= prev[1] + MERGE_GAP:
                # Gộp nếu chạm hoặc cách nhau dưới ngưỡng MERGE_GAP
                merged_regions[-1] = (prev[0], max(prev[1], current[1]))
            else:
                merged_regions.append(current)

        anomaly_regions = merged_regions

    return anomaly_regions, processed_scores, peaks


def plot_vad_animation(anomaly_scores, fps=30, save_path="vad_plot.mp4", 
                      high_threshold=None, low_threshold=None,
                      window_length=None, polyorder=None):
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
    total_frames = len(anomaly_scores)
    
    anomaly_regions, processed_scores, peaks = find_anomaly_regions(
        anomaly_scores,
        high_threshold=high_threshold,
        low_threshold=low_threshold,
        window_length=window_length,
        polyorder=polyorder
    )

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

    # Hàm cập nhật cho animation
    def update(frame):
        line.set_data(x_values[:frame + 1], y_values[:frame + 1])
        point.set_data([x_values[frame]], [y_values[frame]])
        return line, point

    ani = FuncAnimation(fig, update, frames=range(0, total_frames, 1), interval=1000 / fps, blit=True)

    writer = FFMpegWriter(fps=min(30, int(fps)), metadata=dict(artist='Video Anomaly Detection'), bitrate=800)
    ani.save(save_path, writer=writer)
    plt.close(fig)

    return save_path