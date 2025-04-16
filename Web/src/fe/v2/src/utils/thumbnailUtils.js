import { baseUrl } from "./constants";
import { storeVideo } from "./indexedDBStorage";
import { generateVideoThumbnails } from "@rajesh896/video-thumbnails-generator";

// Generate and store thumbnail using the library
export const generateAndStoreThumbnail = async (filename, onSuccess, onError) => {
  console.log(`Starting thumbnail generation for: ${filename}`);
  
  if (!filename) {
    console.error("Invalid filename provided for thumbnail generation");
    if (onError) onError("Invalid filename");
    return null;
  }
  
  try {
    // First, fetch the video file as a blob
    const videoUrl = `${baseUrl}apis/video/get_video/${encodeURIComponent(filename)}`;
    console.log(`Fetching video from URL: ${videoUrl}`);
    
    const response = await fetch(videoUrl);
    if (!response.ok) {
      throw new Error(`Failed to fetch video: ${response.status} ${response.statusText}`);
    }
    
    const videoBlob = await response.blob();
    console.log(`Fetched video blob of size: ${videoBlob.size} bytes`);
    
    // Generate 1 thumbnail using the library
    const thumbnails = await generateVideoThumbnails(videoBlob, 1);
    
    if (thumbnails && thumbnails.length > 0) {
      const thumbnailUrl = thumbnails[0];
      console.log(`Thumbnail generated successfully for ${filename}`);
      
      // Store the thumbnail in IndexedDB
      await storeVideo(`thumbnail_${filename}`, null, { thumbnail: thumbnailUrl });
      console.log(`Thumbnail stored for ${filename}`);
      
      // Call the success callback
      if (onSuccess) {
        onSuccess(thumbnailUrl);
      }
      
      return thumbnailUrl;
    } else {
      throw new Error("No thumbnails were generated");
    }
  } catch (error) {
    console.error(`Error generating thumbnail: ${error.message}`);
    if (onError) onError(error.message);
    return null;
  }
};

// Utility to get a fallback thumbnail for errors
export const getFallbackThumbnail = () => {
  return "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='100' height='100' viewBox='0 0 24 24'%3E%3Cpath fill='%23aaa' d='M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z'/%3E%3C/svg%3E";
};