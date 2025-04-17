import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import PillowWriter,FuncAnimation,FFMpegWriter
import time


__all__ = ["plot_vad_animation"]

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



def plot_vad_animation(anomaly_scores, fps=30, save_path="vad_plot.mp4", high_threshold=0.5, low_threshold=0.4):
    """
    Creates an animated plot of video anomaly detection scores with intelligent
    axis rescaling and blitting for improved performance.
    
    Parameters:
    -----------
    anomaly_scores : list or array
        The anomaly scores to be plotted
    fps : float
        Frames per second of the original video
    save_path : str
        Path to save the animation (should end with .mp4)
    high_threshold : float
        Threshold above which anomaly regions are highlighted (default: 0.7)
    low_threshold : float
        Threshold below which anomaly regions stop being highlighted (default: 0.4)
        
    Returns:
    --------
    str
        Path where the animation was saved
    """

    total_frames = len(anomaly_scores)
    
    # Set up figure
    fig, ax = plt.subplots(figsize=(10, 5), tight_layout=True)
    line, = ax.plot([], [], lw=2, label="Anomaly Score", color='blue')
    point, = ax.plot([], [], 'ro', ms=6)  # Current frame marker
    
    ax.set_xlabel("Frames")
    ax.set_ylabel("Anomaly Score")
    ax.set_xlim(0, total_frames)
    ax.set_ylim(0, 1.0)
    ax.legend()
    
    # Highlight anomaly regions
    anomaly_regions = []
    in_anomaly = False
    start = 0
    
    for i, score in enumerate(anomaly_scores):
        if score >= high_threshold and not in_anomaly:
            in_anomaly = True
            start = i
        elif score < high_threshold and in_anomaly:
            in_anomaly = False
            anomaly_regions.append((start, i))
    if in_anomaly:
        anomaly_regions.append((start, total_frames))
    
    for start, end in anomaly_regions:
        ax.axvspan(start, end, color='red', alpha=0.3)
    
    def update(frame):
        x_values = np.arange(frame + 1)
        y_values = anomaly_scores[:frame + 1]
        line.set_data(x_values, y_values)
        point.set_data([frame], [anomaly_scores[frame]])
        return line, point
    
    ani = FuncAnimation(fig, update, frames=range(total_frames), interval=30, blit=True)
    
    writer = FFMpegWriter(fps=min(30, int(fps)), metadata=dict(artist='Video Anomaly Detection'), bitrate=2000)
    ani.save(save_path, writer=writer)
    plt.close(fig)
    
    return save_path