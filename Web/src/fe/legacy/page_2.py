import os
import requests
from pathlib import Path


import safehttpx
import gradio as gr
from gradio_modal import Modal


from ...api.config import API_URL

__all__ = ["page_2"]


async def _async_validate_url_override(hostname: str):
    if hostname in ["localhost", "127.0.0.1"]:
        return hostname
    return await _orig_validate(hostname)


_orig_validate = safehttpx.async_validate_url
safehttpx.async_validate_url = _async_validate_url_override


def get_uploaded_videos():
    """Lấy danh sách video đã upload"""
    try:
        response = requests.get(f"{API_URL}/apis/video/get_all_video")
        if response.status_code == 200:
            videos = response.json().get("videos", [])
            # Return a list of tuples (image_path/url, caption) for the Gallery
            return [
                (f"{API_URL}/apis/video/get_video/{video['filename']}", video['filename'])
                for video in videos
            ]
        else:
            print(f"Error: Could not retrieve video list. Status code: {response.status_code}")
            return []
    except Exception as e:
        print(f"Error loading videos: {e}")
        return []


def get_anomaly_plot(video_filename: str):
    """Lấy anomaly plot từ API"""
    try:
        # Get plot directly from API instead of local path
        plot_url = f"{API_URL}/apis/video/get_plot/{video_filename}"
        
        # Verify if plot exists by making a HEAD request
        response = requests.head(plot_url)
        if response.status_code == 200:
            return plot_url
        else:
            print(f"Plot not found at {plot_url}, status code: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error checking plot: {e}")
        return None


def open_modal(evt: gr.SelectData):
    """Xử lý khi user click vào video trong Gallery"""
    try:
        index = evt.index
        print(f"Selected index: {index}")
        print(f"evt.value: {evt.value}")
        print(f"type(evt.value): {type(evt.value)}")

        # Extract filename from the dictionary evt.value
        filename = evt.value.get('caption')
        if not filename:
            print("Error: Filename not found in evt.value")
            return (
                gr.update(value=None),
                gr.update(value=None),
                gr.update(visible=False),
            )

        video_url = f"{API_URL}/apis/video/get_video/{filename}"
        print(f"Selected Video URL: {video_url}")

        if not video_url:
            print("Error: Video URL is empty.")
            return (
                gr.update(value=None),
                gr.update(value=None),
                gr.update(visible=False),
            )

        plot_url = get_anomaly_plot(filename)
        print(f"Plot URL: {plot_url}")

        return (
            gr.update(value=video_url),
            gr.update(value=plot_url),
            gr.update(visible=True),
        )
    except Exception as e:
        print(f"Error in open_modal: {e}")
        return (
            gr.update(value=None),
            gr.update(value=None),
            gr.update(visible=False),
        )


with gr.Blocks() as page_2:
    gr.Markdown("# 📂 List of Videos")

    # Create a gallery to display videos
    gallery = gr.Gallery(
        label="📌 Danh sách video",
        columns=[5],
        rows=[2],
        object_fit="contain",
        height="300",
        allow_preview=False,
        show_label=True,
        elem_id="video-gallery"
    )

    # Create modal to show video and plot
    with Modal(visible=False, allow_user_close=True) as video_modal:
        with gr.Row():
            with gr.Column(scale=1):
                video_display = gr.Video(
                    label="📺 Video Preview",
                    height=500,
                    autoplay=True,
                    include_audio=False,
                )
            with gr.Column(scale=1):
                plot_display = gr.Video(
                    label="📊 Anomaly Plot",
                    height=500,
                    autoplay=True,
                )

    # When a video is clicked, show it in the modal
    gallery.select(
        fn=open_modal,
        inputs=None,
        outputs=[video_display, plot_display, video_modal]
    )

    # Load videos when the page loads
    page_2.load(get_uploaded_videos, outputs=[gallery])
