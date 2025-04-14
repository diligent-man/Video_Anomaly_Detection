import apiHelper from "../utils/apiHelper.js";
import { apiUrls, baseUrl } from "../utils/constants.js";

/**
 * Upload a new video file to the server
 * @param {Object} params - Parameters
 * @param {File} params.videoFile - The video file to upload
 * @param {Function} params.onProgress - Progress callback function (optional)
 * @param {Function} params.onSuccess - Success callback function
 * @param {Function} params.onFail - Failure callback function
 */
export async function uploadVideoApi({
    videoFile,
    onProgress,
    onSuccess,
    onFail,
  }) {
    try {
      if (!videoFile) {
        onFail("No video file provided");
        return;
      }
  
      // Create form data to send file - make sure to use the correct field name
      const formData = new FormData();
      formData.append("file", videoFile); // Using "file" parameter as expected by backend
  
      // Configuration for progress tracking
      const config = onProgress 
        ? { 
            onUploadProgress: (progressEvent) => {
              const percentCompleted = Math.round((progressEvent.loaded * 100) / progressEvent.total);
              onProgress(percentCompleted);
            },
            headers: {
              'Content-Type': 'multipart/form-data',
            }
          } 
        : {
            headers: {
              'Content-Type': 'multipart/form-data',
            }
          };
  
      const response = await apiHelper.postFormData(apiUrls.UploadVideo, formData, config);
  
      // Handle response
      if (response && (response.filename || response.status === "processed" || response.status === "existing")) {
        // Ensure plot_path is correctly formatted with full URL
        if (response.plot_path) {
          // Check if plot_path is already a full URL
          if (!response.plot_path.startsWith('http')) {
            response.plot_path = `${baseUrl}apis/video/get_plot/${encodeURIComponent(response.filename)}`;
          }
        }
        onSuccess(response);
      } else {
        onFail(response?.message || "Failed to upload video");
      }
    } catch (error) {
      console.error("Upload video error:", error);
      onFail(error?.message || "An error occurred while uploading the video");
    }
  }

/**
 * Get a specific video by name
 * @param {string} videoName - Name of the video to retrieve
 * @returns {Promise<Object>} Video data
 */
export async function getVideoApi(videoName) {
  try {
    if (!videoName) {
      throw new Error("Video name is required");
    }

    const response = await apiHelper.get(apiUrls.getVideo(videoName));
    
    // The backend returns a FileResponse, so this should be a binary stream
    return response;
  } catch (error) {
    // Handle specific HTTP errors
    if (error.response) {
      if (error.response.status === 404) {
        throw new Error(`Video with name ${videoName} not found`);
      }
      throw new Error(error.response.data?.message || 'Failed to retrieve video');
    }
    
    console.error("Get video error:", {
      error: error.message,
      videoName,
      response: error.response
    });
    throw error;
  }
}

/**
 * Get all videos from the server
 * @returns {Promise<Array>} Array of video objects
 */
export async function getAllVideosApi() {
  try {
    const response = await apiHelper.get(apiUrls.GetAllVideos);
    
    // Make sure we return an array of videos
    if (response && response.videos && Array.isArray(response.videos)) {
      return response.videos;
    } else {
      console.error("Unexpected API response format:", response);
      return [];
    }
  } catch (error) {
    console.error("Error fetching videos:", error);
    throw error;
  }
}

/**
 * Delete a video by name
 * @param {string} videoName - Name of the video to delete
 * @returns {Promise<Object>} Response object with deletion status
 */
export async function deleteVideoApi(videoName) {
  try {
    if (!videoName) {
      throw new Error("Video name is required");
    }

    const response = await apiHelper.delete(apiUrls.DeleteVideo(videoName));
    
    // Check for success
    if (response && response.message && response.message.includes("deleted successfully")) {
      return {
        success: true,
        videoName,
        deleted: response.deleted || ["video"],
        message: `Video ${videoName} deleted successfully`
      };
    }
    
    throw new Error(response?.message || "Failed to delete video");
  } catch (error) {
    console.error("Delete video error:", error);
    throw new Error(error?.message || "An error occurred while deleting the video");
  }
}

/**
 * Get anomaly score for a video
 * @param {string} videoName - Name of the video
 * @returns {Promise<Object>} Anomaly score data
 */
export async function getScoreApi(videoName) {
  try {
    if (!videoName) {
      throw new Error("Video name is required");
    }

    const response = await apiHelper.get(apiUrls.GetScore(videoName));
    
    if (!response || !response.scores) {
      throw new Error('Invalid response format');
    }
    
    // Add plot URL if not present but available
    if (response.plot_path && !response.plot_path.startsWith('http')) {
      response.plot_path = `${baseUrl}apis/video/get_plot/${encodeURIComponent(videoName)}`;
    }
    
    return response;
  } catch (error) {
    console.error("Get score error:", {
      error: error.message,
      videoName,
      response: error.response
    });
    throw error;
  }
}

/**
 * Check if score is available for a video
 * @param {string} videoName - Name of the video
 * @returns {Promise<Object>} Score availability information
 */
export async function checkScoreApi(videoName) {
  try {
    if (!videoName) {
      throw new Error("Video name is required");
    }

    const response = await apiHelper.get(apiUrls.CheckScore(videoName));
    
    return {
      scoreAvailable: response?.scores_exist || false,
      plotAvailable: response?.plot_exist || false,
      scoresPath: response?.scores_path,
      plotPath: response?.plot_exist 
        ? `${baseUrl}apis/video/get_plot/${encodeURIComponent(videoName)}` 
        : null
    };
  } catch (error) {
    console.error("Check score error:", {
      error: error.message,
      videoName,
    });
    return { scoreAvailable: false, plotAvailable: false };
  }
}

/**
 * Get plot animation for a video
 * @param {string} videoName - Name of the video
 * @returns {string} URL to the plot animation
 */
export function getPlotUrl(videoName) {
  if (!videoName) return null;
  return `${baseUrl}apis/video/get_plot/${encodeURIComponent(videoName)}`;
}