from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
from urllib.parse import unquote

import os
import numpy as np
from datetime import datetime

from ..utils.video_utils import get_video_info, plot_vad_animation
from ..inference.score_saver import run_vad_model as model_run_vad


router = APIRouter(prefix="/apis/video", tags=["Video"])

# Định nghĩa thư mục lưu trữ
TMP_DIR = Path("tmp").resolve()
VIDEO_DIR = TMP_DIR / "videos"
SCORES_DIR = TMP_DIR / "scores"
PLOTS_DIR = TMP_DIR / "plots"

# Tạo các thư mục nếu chưa tồn tại
os.makedirs(VIDEO_DIR, exist_ok=True)
os.makedirs(SCORES_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)


def run_vad_model(video_path, total_frames):
    """ Run the actual VAD model and save anomaly scores """
    return model_run_vad(video_path, total_frames, SCORES_DIR)


def get_scores_path(video_name):
    """Generate the expected scores file path for a video"""
    return SCORES_DIR / f"{Path(video_name).stem}_scores.npy"


def get_plot_path(video_name):
    """Generate the expected plot animation file path for a video"""
    return PLOTS_DIR / f"{Path(video_name).stem}_plot.mp4"


def generate_plot(video_name, scores=None):
    """Generate plot animation for a video based on its anomaly scores"""
    try:
        plot_path = get_plot_path(video_name)
        
        # If scores not provided, load them
        if scores is None:
            scores_file = get_scores_path(video_name)
            if not scores_file.exists():
                return None
            scores = np.load(scores_file).tolist()
        
        # Get FPS from video
        video_path = VIDEO_DIR / video_name
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

@router.get("/")
def say_hello():
    return {
        "exit code": 200,
        "message": "Video Anomaly Detection API"
    }

@router.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload video, tính toán anomaly scores và tạo plot animation
    """
    try:
        CHUNK_SIZE = 1024*1024  # 1MB chunks
        video_path = VIDEO_DIR / file.filename
        scores_file = get_scores_path(file.filename)
        plot_file = get_plot_path(file.filename)
        
        # Get current date and time for upload_date
        upload_date = datetime.now().isoformat()
        
        result = {
            "filename": file.filename,
            "video_path": str(video_path),
            "upload_date": upload_date
        }
        
        # Lưu video
        with open(video_path, "wb") as buffer:
            while chunk := await file.read(CHUNK_SIZE):
                buffer.write(chunk)
        
        # Tính toán scores nếu chưa có
        if not scores_file.exists():
            _, total_frames, _ = get_video_info(video_path)
            scores_file_path = run_vad_model(video_path, total_frames)
            result["scores_path"] = str(scores_file_path)
            result["status"] = "processed"
            
            # Đọc scores vừa tạo để tạo plot
            scores = np.load(scores_file).tolist()
        else:
            # Nếu scores đã tồn tại
            result["scores_path"] = str(scores_file)
            result["status"] = "existing"
            scores = np.load(scores_file).tolist()
        
        # Tạo plot animation nếu chưa có hoặc cần tạo lại
        if not plot_file.exists() or result["status"] == "processed":
            plot_path = generate_plot(file.filename, scores)
            if plot_path:
                result["plot_path"] = plot_path
            else:
                result["plot_status"] = "failed"
        else:
            result["plot_path"] = str(plot_file)
            result["plot_status"] = "existing"
        
        # Thêm CORS header vào response
        response = JSONResponse(content=result)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
    
    except Exception as e:
        print(f"Error in upload_video: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"message": f"Error uploading video: {str(e)}"}, 
            headers={"Access-Control-Allow-Origin": "*"}
        )

@router.get("/get_video/{video_name}")
async def get_video(video_name: str):
    """ Lấy video theo tên """
    decoded_filename = unquote(video_name)
    file_path = VIDEO_DIR / decoded_filename

    if not file_path.exists():
        return JSONResponse(status_code=404, content={"message": "File not found"})

    # Create the FileResponse
    response = FileResponse(file_path, media_type="video/mp4")

    # *** ADD THIS LINE ***
    # Manually add the CORS header to allow access from any origin
    response.headers["Access-Control-Allow-Origin"] = "*"

    return response # Return the response with the added header

@router.get("/get_all_video")
async def list_videos():
    """ Liệt kê danh sách video """
    try:
        videos = []
        for file in VIDEO_DIR.glob("*"):
            if file.suffix.lower() in ['.mp4', '.avi', '.mov', '.mkv']:
                # Use file creation time as upload date
                upload_date = datetime.fromtimestamp(file.stat().st_ctime).isoformat()
                
                video_info = {
                    "filename": file.name,
                    "path": str(file),
                    "size": file.stat().st_size,
                    "has_scores": get_scores_path(file.name).exists(),
                    "has_plot": get_plot_path(file.name).exists(),
                    "upload_date": upload_date
                }
                
                videos.append(video_info)
                
        # Sort videos by upload date (newest first)
        videos.sort(key=lambda x: x["upload_date"], reverse=True)
                
        return JSONResponse(status_code=200, content={"videos": videos})
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error listing videos: {str(e)}"})

@router.delete("/delete_video/{video_name}")
async def delete_video(video_name: str):
    """ Xóa video và tất cả dữ liệu liên quan (scores, plot) """
    try:
        video_path = VIDEO_DIR / video_name
        scores_file = get_scores_path(video_name)
        plot_file = get_plot_path(video_name)
        
        deleted = []
        status_code = 200
        
        # Kiểm tra xem video có tồn tại không
        if not video_path.exists():
            return JSONResponse(
                status_code=404, 
                content={"message": f"Video {video_name} not found"}
            )
        
        # Xóa video
        try:
            if video_path.is_file():
                video_path.unlink()
                deleted.append("video")
            else:
                return JSONResponse(
                    status_code=400, 
                    content={"message": f"{video_name} is not a file"}
                )
        except Exception as e:
            return JSONResponse(
                status_code=500, 
                content={"message": f"Error deleting video file: {str(e)}"}
            )
        
        # Xóa scores nếu tồn tại
        try:
            if scores_file.exists():
                scores_file.unlink()
                deleted.append("scores")
        except Exception as e:
            # Không dừng lại nếu xóa scores thất bại
            print(f"Error deleting scores file: {str(e)}")
            
        # Xóa plot nếu tồn tại
        try:
            if plot_file.exists():
                plot_file.unlink()
                deleted.append("plot")
        except Exception as e:
            # Không dừng lại nếu xóa plot thất bại
            print(f"Error deleting plot file: {str(e)}")
        
        # Kiểm tra kết quả xóa
        if not deleted:
            status_code = 500
            message = "Failed to delete any files"
        elif "video" in deleted:
            message = f"Video {video_name} and related files deleted successfully"
        else:
            status_code = 206  # Partial Content
            message = f"Some files for {video_name} were deleted, but not the video itself"
            
        return JSONResponse(
            status_code=status_code, 
            content={
                "message": message, 
                "deleted": deleted,
                "video_name": video_name
            }
        )
    except PermissionError:
        return JSONResponse(
            status_code=500, 
            content={
                "message": f"Permission denied to delete {video_name}", 
                "video_name": video_name
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={
                "message": f"Error deleting video: {str(e)}", 
                "video_name": video_name
            }
        )

@router.get("/check_score/{video_name}")
async def check_scores_exist(video_name: str):
    """Check if scores for a specific video already exist"""
    scores_file = get_scores_path(video_name)
    plot_file = get_plot_path(video_name)
    
    return {
        "video_name": video_name,
        "scores_exist": scores_file.exists(),
        "scores_path": str(scores_file) if scores_file.exists() else None,
        "plot_exist": plot_file.exists(),
        "plot_path": str(plot_file) if plot_file.exists() else None
    }

@router.get("/get_score/{video_name}")
async def get_anomaly_scores(video_name: str):
    """ Lấy anomaly scores theo tên video """
    scores_file = get_scores_path(video_name)
    
    if not scores_file.exists():
        # If scores don't exist but video does, process it
        video_path = VIDEO_DIR / video_name
        if video_path.exists():
            _, total_frames, _ = get_video_info(video_path)
            run_vad_model(video_path, total_frames)
            
            if scores_file.exists():
                scores = np.load(scores_file).tolist()
                
                # Generate plot animation
                plot_path = generate_plot(video_name, scores)
                
                return JSONResponse(
                    status_code=200, 
                    content={
                        "scores": scores, 
                        "status": "newly_processed",
                        "plot_path": plot_path
                    }
                )
        
        return JSONResponse(status_code=404, content={"message": "Scores file not found"})
    
    try:
        scores = np.load(scores_file).tolist()
        
        # Check if plot exists; if not, generate it
        plot_path = get_plot_path(video_name)
        if not plot_path.exists():
            plot_path_str = generate_plot(video_name, scores)
        else:
            plot_path_str = str(plot_path)
        
        return JSONResponse(
            status_code=200, 
            content={
                "scores": scores, 
                "status": "existing",
                "plot_path": plot_path_str
            }
        )
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error reading scores: {str(e)}"})

@router.api_route("/get_plot/{video_name}", methods=["GET", "HEAD"])
async def get_plot(video_name: str):
    """ Lấy file plot animation theo tên video """
    plot_path = get_plot_path(video_name)

    if not plot_path.exists():
        # Nếu plot chưa tồn tại, kiểm tra xem scores có không để tạo plot
        scores_file = get_scores_path(video_name)
        if scores_file.exists():
            try:
                scores = np.load(scores_file).tolist()
                generate_plot(video_name, scores)

                if plot_path.exists():
                    # Create the FileResponse
                    response = FileResponse(plot_path, media_type="video/mp4")
                    # *** ADD THIS LINE ***
                    response.headers["Access-Control-Allow-Origin"] = "*"
                    return response
            except Exception as e:
                 print(f"Error generating or serving plot after creation: {e}")
                 # Fall through to the not found response if generation failed here

        return JSONResponse(status_code=404, content={"message": "Plot not found and could not be generated"})

    # Create the FileResponse for existing plot
    response = FileResponse(plot_path, media_type="video/mp4")

    # *** ADD THIS LINE ***
    # Manually add the CORS header to allow access from any origin
    response.headers["Access-Control-Allow-Origin"] = "*"

    return response # Return the response with the added header