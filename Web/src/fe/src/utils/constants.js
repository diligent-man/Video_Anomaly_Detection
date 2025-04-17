export const baseUrl = "http://localhost:8000/";

export const apiUrls = {
  //Video 
  UploadVideo: "apis/video/upload_video",
  getVideo: (video_name) => `apis/video/get_video/${video_name}`,
  GetAllVideos: "apis/video/get_all_video", // Changed from get_all_videos to get_all_video to match backend
  DeleteVideo: (video_name) => `apis/video/delete_video/${video_name}`,

  //Score
  GetScore: (video_name) => `apis/video/get_score/${video_name}`,
  CheckScore: (video_name) => `apis/video/check_score/${video_name}`,
  
  //Plot
  GetPlot: (video_name) => `apis/video/get_plot/${video_name}`, // Added endpoint for plot retrieval
}