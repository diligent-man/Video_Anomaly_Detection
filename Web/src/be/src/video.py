from fastapi import APIRouter, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from pathlib import Path
from urllib.parse import unquote

import numpy as np
from datetime import datetime

from .utils.handle_be import (
    run_vad_model,
    get_scores_path,
    get_plot_path,
    generate_plot,
    VIDEO_DIR,
    SCORES_DIR,
    PLOTS_DIR
)

router = APIRouter(prefix="/apis/video", tags=["Video"])



@router.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload video, tính toán anomaly scores và tạo plot animation
    """
    print(f"[UPLOAD] Starting upload process for file: {file.filename}")
    try:
        CHUNK_SIZE = 1024*1024  # 1MB chunks
        video_path = VIDEO_DIR / file.filename
        scores_file = get_scores_path(file.filename)
        plot_file = get_plot_path(file.filename)
        
        print(f"[UPLOAD] Paths prepared: video={video_path}, scores={scores_file}, plot={plot_file}")
        
        # Get current date and time for upload_date
        upload_date = datetime.now().isoformat()
        
        result = {
            "filename": file.filename,
            "video_path": str(video_path),
            "upload_date": upload_date
        }
        
        # Lưu video
        print(f"[UPLOAD] Starting to save video file {file.filename}")
        with open(video_path, "wb") as buffer:
            chunk_count = 0
            while chunk := await file.read(CHUNK_SIZE):
                buffer.write(chunk)
                chunk_count += 1
                if chunk_count % 10 == 0:  # Log every 10MB
                    print(f"[UPLOAD] Written {chunk_count * CHUNK_SIZE / 1024 / 1024:.1f}MB of video data")
        print(f"[UPLOAD] Video saved successfully: {video_path}")
        
        # Tính toán scores nếu chưa có
        video_fps = None  # Khởi tạo biến fps
        print(f"[UPLOAD] Checking for existing scores at {scores_file}")
        if not scores_file.exists():
            print(f"[UPLOAD] No existing scores found, generating new scores")
            scores_file_path, video_fps = run_vad_model(str(video_path))
            print(f"[UPLOAD] Scores generated successfully: path={scores_file_path}, fps={video_fps}")
            result["scores_path"] = scores_file_path
            result["fps"] = video_fps  # Lưu fps vào kết quả
            result["status"] = "processed"
            
            # Đọc scores vừa tạo để tạo plot
            print(f"[UPLOAD] Loading newly created scores from {scores_file}")
            scores = np.load(scores_file).tolist()
            print(f"[UPLOAD] Scores loaded: {len(scores)} data points")
            
            # Tạo mới plot khi scores mới được tạo
            print(f"[UPLOAD] Starting plot generation for {file.filename}")
            plot_path = generate_plot(file.filename, scores, fps=video_fps)
            if plot_path:
                print(f"[UPLOAD] Plot generated successfully: {plot_path}")
                result["plot_path"] = plot_path
                result["plot_status"] = "generated"
            else:
                print(f"[UPLOAD] Plot generation failed")
                result["plot_status"] = "failed"
        else:
            # Nếu scores đã tồn tại
            print(f"[UPLOAD] Using existing scores from {scores_file}")
            result["scores_path"] = str(scores_file)
            result["status"] = "existing"
            
            # Kiểm tra xem plot đã tồn tại chưa
            if plot_file.exists():
                # Xóa plot đã có
                print(f"[UPLOAD] Existing plot file found, deleting: {plot_file}")
                try:
                    plot_file.unlink()
                    print(f"[UPLOAD] Existing plot file deleted successfully")
                except Exception as e:
                    print(f"[UPLOAD] Warning: Could not delete existing plot file: {str(e)}")

            # Tạo plot mới
            print(f"[UPLOAD] Generating new plot")
            scores = np.load(scores_file).tolist()
            print(f"[UPLOAD] Scores loaded: {len(scores)} data points")

            # Lấy fps từ video nếu không có trong scores
            if video_fps is None:
                try:
                    from .utils.video_utils import get_video_info
                    video_fps, _, _ = get_video_info(str(video_path))
                    result["fps"] = video_fps
                except Exception as e:
                    print(f"[UPLOAD] Warning: Could not get video FPS: {str(e)}")

            plot_path = generate_plot(file.filename, scores, fps=video_fps)
            if plot_path:
                print(f"[UPLOAD] Plot generated successfully: {plot_path}")
                result["plot_path"] = plot_path
                result["plot_status"] = "generated"
            else:
                print(f"[UPLOAD] Plot generation failed")
                result["plot_status"] = "failed"
        # Thêm CORS header vào response
        print(f"[UPLOAD] Process completed successfully, returning response")
        response = JSONResponse(content=result)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response
    
    except Exception as e:
        print(f"[UPLOAD] Error in upload_video: {str(e)}")
        import traceback
        print(f"[UPLOAD] Stacktrace: {traceback.format_exc()}")
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
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

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
            scores_file_path = run_vad_model(str(video_path))
            
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
                    response = FileResponse(plot_path, media_type="video/mp4")
                    response.headers["Access-Control-Allow-Origin"] = "*"
                    return response
            except Exception as e:
                 print(f"Error generating or serving plot after creation: {e}")

        return JSONResponse(status_code=404, content={"message": "Plot not found and could not be generated"})

    response = FileResponse(
        plot_path, 
        media_type="video/mp4",
        headers={
            "Access-Control-Allow-Origin": "*",
            "Accept-Ranges": "bytes"  # Important for large video files
        }
    )
    return response