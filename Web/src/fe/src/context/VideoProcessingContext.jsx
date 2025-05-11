import React, { createContext, useState, useEffect, useRef } from 'react';
import { useLocalStorage } from "@mantine/hooks";
import { uploadVideoApi, getPlotUrl } from "../apis/Video";
import { storeVideo, getVideo, clearVideo } from "../utils/indexedDBStorage";
import { baseUrl } from "../utils/constants";
import { notifications } from "@mantine/notifications";
import { IconAlertCircle, IconCircleCheck } from "@tabler/icons-react";

export const VideoProcessingContext = createContext();

export function VideoProcessingProvider({ children }) {
  // LocalStorage for persistence between refreshes
  const [storedVideoName, setStoredVideoName] = useLocalStorage({
    key: "anomaly-detection-video-name",
    defaultValue: null,
  });

  const [storedPlotUrl, setStoredPlotUrl] = useLocalStorage({
    key: "anomaly-detection-plot-url",
    defaultValue: null,
  });

  // State for current processing
  const [video, setVideo] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [plotVideo, setPlotVideo] = useState(storedPlotUrl);

  // Refs
  const videoNameRef = useRef(storedVideoName);
  const syncCleanupRef = useRef(null);

  // Restore video from IndexedDB if available
  useEffect(() => {
    const restoreVideo = async () => {
      if (storedVideoName) {
        try {
          setLoading(true);
          const videoData = await getVideo("current-video");

          if (videoData && videoData.blob) {
            // Create a File object from the stored blob
            const file = new File([videoData.blob], storedVideoName, {
              type: videoData.metadata?.type || "video/mp4",
            });

            setVideo(file);

            // Create a URL for the video visualization
            const url = URL.createObjectURL(file);
            setVideoUrl(url);

            // If we have a plot URL stored, also restore plot video state
            if (storedPlotUrl) {
              setPlotVideo(storedPlotUrl);
              setSuccess(true); // Show success message for restored analysis
            }
          } else {
            console.warn("Video data not found in IndexedDB");
            // Clear localStorage if IndexedDB data is missing
            setStoredVideoName(null);
            setStoredPlotUrl(null);
          }
        } catch (error) {
          console.error("Failed to restore video from IndexedDB:", error);
          // Clear invalid storage data
          setStoredVideoName(null);
          setStoredPlotUrl(null);
        } finally {
          setLoading(false);
        }
      }
    };

    restoreVideo();

    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [storedVideoName, storedPlotUrl]);

  // Handle video selection
  const handleVideoSelect = async (selectedVideo) => {
    if (selectedVideo) {
      // Clean up previous object URL if exists
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }

      setVideo(selectedVideo);
      setPlotVideo(null);
      setError(null);
      setSuccess(false);

      // Create a URL for the video visualization
      const url = URL.createObjectURL(selectedVideo);
      setVideoUrl(url);

      // Store video in IndexedDB
      try {
        await storeVideo("current-video", selectedVideo, {
          name: selectedVideo.name,
          type: selectedVideo.type,
          size: selectedVideo.size,
        });

        // Store metadata in localStorage for quick checks
        setStoredVideoName(selectedVideo.name);

        // Clear stored plot URL since we have a new video
        setStoredPlotUrl(null);

        videoNameRef.current = selectedVideo.name;

        notifications.show({
          title: "Video Uploaded",
          message: `${selectedVideo.name} is ready for analysis`,
          color: "blue",
          autoClose: 3000,
        });
      } catch (error) {
        console.error("Failed to store video in IndexedDB:", error);
        setError(
          "Failed to save video locally. Your browser may not support this feature or the video may be too large."
        );
      }
    } else {
      // Clear everything when video is cleared
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }

      setVideo(null);
      setVideoUrl(null);
      setPlotVideo(null);
      setError(null);
      setSuccess(false);

      // Clear from IndexedDB
      try {
        await clearVideo("current-video");
      } catch (err) {
        console.error("Error clearing video from IndexedDB:", err);
      }

      // Clear from localStorage
      setStoredVideoName(null);
      setStoredPlotUrl(null);

      videoNameRef.current = null;
    }
  };

  // Process video function
  const processVideo = () => {
    if (!video) return;

    setProcessing(true);
    setError(null);
    setSuccess(false);
    setLoading(true);

    // Show processing notification
    const processingNotificationId = notifications.show({
      title: "Processing",
      message: "Your video is being analyzed for anomalies...",
      color: "blue",
      loading: true,
      autoClose: false,
      withCloseButton: false,
    });

    uploadVideoApi({
      videoFile: video,
      onProgress: (percent) => console.log(`Upload progress: ${percent}%`),
      onSuccess: (data) => {
        setSuccess(true);
        setProcessing(false);
        setLoading(false);

        // Close the processing notification
        notifications.update({
          id: processingNotificationId,
          title: "Analysis Complete",
          message: "Your video has been successfully analyzed!",
          color: "green",
          icon: <IconCircleCheck />,
          loading: false,
          autoClose: 5000,
        });

        let plotUrl = null;

        // Handle plot path correctly
        if (data.plot_path) {
          // Check if plot_path is already a full URL
          if (data.plot_path.startsWith("http")) {
            plotUrl = data.plot_path;
          } else {
            // Otherwise, use the baseUrl from constants
            plotUrl = `${baseUrl}${data.plot_path}`;
          }
        } else if (data.filename) {
          // If no plot_path provided but we have the filename, use the getPlotUrl helper
          plotUrl = getPlotUrl(data.filename);
        }

        setPlotVideo(plotUrl);
        // Store plot URL in localStorage for persistence
        setStoredPlotUrl(plotUrl);
      },
      onFail: (errorMsg) => {
        setError(`Processing failed: ${errorMsg}`);
        setProcessing(false);
        setLoading(false);

        // Update the notification for failure
        notifications.update({
          id: processingNotificationId,
          title: "Processing Failed",
          message: errorMsg,
          color: "red",
          icon: <IconAlertCircle />,
          loading: false,
          autoClose: 5000,
        });
      },
    });
  };

  // Reset function
  const handleReset = () => {
    if (videoUrl) {
      URL.revokeObjectURL(videoUrl);
    }

    setPlotVideo(null);
    setError(null);
    setSuccess(false);
    setStoredPlotUrl(null);

    notifications.show({
      title: "Reset Complete",
      message: "You can now run a new analysis on your video",
      color: "blue",
      autoClose: 3000,
    });
  };

  return (
    <VideoProcessingContext.Provider value={{
      video, 
      videoUrl,
      processing,
      loading,
      error,
      success, 
      plotVideo,
      storedVideoName,
      storedPlotUrl,
      setVideo,
      setVideoUrl,
      setPlotVideo,
      processVideo,
      handleVideoSelect,
      handleReset
    }}>
      {children}
    </VideoProcessingContext.Provider>
  );
}