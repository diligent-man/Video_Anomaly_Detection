import gradio as gr
import requests
from pathlib import Path
from ...api.config import API_URL
from ...utils.video_utils import plot_vad_animation

__all__ = ["page_1"]


def get_uploaded_videos():
    """Get list of uploaded videos"""
    try:
        response = requests.get(f"{API_URL}/apis/video/get_all_video")
        if response.status_code == 200:
            videos = response.json().get("videos", [])
            return [f"{API_URL}/apis/video/get_video/{video['filename']}" for video in videos] if videos else []
        return []
    except Exception as e:
        print(f"Error fetching videos: {str(e)}")
        return []


def process_video(video):
    """Upload video, lấy anomaly scores, vẽ plot"""
    if not video:
        return None, None
    
    try:
        video_path = Path(video)
        with video_path.open("rb") as f:
            files = {"file": (video_path.name, f, "video/mp4")}
            response = requests.post(f"{API_URL}/apis/video/upload_video", files=files)
        
        if response.status_code != 200:
            print(f"Error uploading video: {response.status_code} - {response.text}")
            return None, None
        
        response_data = response.json()
        
        # Check if plot was created
        if "plot_path" in response_data:
            # Use the plot path from the API
            plot_url = f"{API_URL}/apis/video/get_plot/{video_path.name}"
            return video, plot_url
        
        # If no plot was created, try to get scores and create plot manually
        scores_response = requests.get(f"{API_URL}/apis/video/get_score/{video_path.name}")
        if scores_response.status_code != 200:
            print(f"Error getting scores: {scores_response.status_code} - {scores_response.text}")
            return video, None
    
        # Get plot from the API
        plot_url = f"{API_URL}/apis/video/get_plot/{video_path.name}"
        return video, plot_url
            
    except Exception as e:
        print(f"Error in process_video: {str(e)}")
        return video, None


with gr.Blocks() as page_1:
    gr.Markdown("# Video Anomaly Detection")
    
    with gr.Row():
        with gr.Column(scale=1):
            input_video = gr.Video(label="Input Video", height=400, autoplay=True, loop=True)
        with gr.Column(scale=1):
            anomaly_plot = gr.Video(label="Anomaly Plot", height=400, interactive=False, autoplay=True, loop=True)
    
    # Add a dropdown to select from previously uploaded videos
    video_dropdown = gr.Dropdown(
        label="Select Previously Uploaded Video",
        choices=get_uploaded_videos(),
        interactive=True,
        every=30  # Refresh every 30 seconds
    )
    
    submit_btn = gr.Button("Detect Anomaly")
    
    # When video is selected from dropdown, update the input video
    video_dropdown.change(
        fn=lambda x: x,
        inputs=video_dropdown,
        outputs=input_video
    )
    
    # When submit button is clicked, process the video
    submit_btn.click(
        fn=process_video,
        inputs=input_video,
        outputs=[input_video, anomaly_plot]
    )
