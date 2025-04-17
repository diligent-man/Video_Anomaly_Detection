import { Button, Flex, Title, Box, Group, rem } from "@mantine/core";
import { useState, useRef, useContext, useEffect } from "react";
import { useLocalStorage } from "@mantine/hooks";
import {
  IconPlayerPlay,
  IconAlertCircle,
  IconCheck,
  IconRefresh,
  IconCircleCheck,
} from "@tabler/icons-react";
import { notifications } from "@mantine/notifications";
import HeadingLayout from "../../components/Layout/HeadingLayout.jsx";
import VideoUpload from "../../components/VideoUpload/VideoUpload.jsx";
import AnomalyPlot from "../../components/AnomalyPlot/AnomalyPlot.jsx";
import { uploadVideoApi, getPlotUrl } from "../../apis/Video.js";
import { baseUrl } from "../../utils/constants.js";
import { NavbarContext } from "../../context/NavbarContext.jsx";
import { storeVideo, getVideo, clearVideo } from "../../utils/indexedDBStorage.js";
import { synchronizeVideos } from "../../utils/SynchronizeVideo";

export default function HomePage() {
  // Keep metadata in localStorage for quick checks
  const [storedVideoName, setStoredVideoName] = useLocalStorage({
    key: "anomaly-detection-video-name",
    defaultValue: null,
  });

  // Store plot URL in localStorage
  const [storedPlotUrl, setStoredPlotUrl] = useLocalStorage({
    key: "anomaly-detection-plot-url",
    defaultValue: null,
  });

  const [video, setVideo] = useState(null);
  const [videoUrl, setVideoUrl] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [success, setSuccess] = useState(false);
  const [plotVideo, setPlotVideo] = useState(storedPlotUrl);
  const [loading, setLoading] = useState(false);
  const { navbarOpened } = useContext(NavbarContext);

  const videoNameRef = useRef(storedVideoName);

  // Add refs for video synchronization
  const mainVideoRef = useRef(null);
  const plotVideoRef = useRef(null);
  const syncCleanupRef = useRef(null);

  // Common style objects without animation
  const processButtonStyle = {
    transition: "all 0.3s ease",
    position: "relative",
    overflow: "hidden",
    fontWeight: 600,
    transform: "translateY(0)",
  };

  const readyButtonStyle = {
    background: "linear-gradient(45deg, #228be6, #4dabf7)",
  };

  const completeButtonStyle = {
    background: "linear-gradient(45deg, #40c057, #82c91e)",
  };

  // Adjust component size based on navbar state
  const baseWidth = 580;
  const expandedWidth = navbarOpened ? baseWidth : 600;
  const componentHeight = 500;
  const transitionDuration = "0.7s";

  // Add leftPadding adjustment based on navbar state
  const leftPadding = navbarOpened ? "0px" : "20px";

  // Show error notification when error state updates
  useEffect(() => {
    if (error) {
      notifications.show({
        title: "Error",
        message: error,
        color: "red",
        icon: <IconAlertCircle />,
        autoClose: 5000,
      });
    }
  }, [error]);

  // Show success notification when success state updates
  useEffect(() => {
    if (success) {
      notifications.show({
        title: "Success",
        message: "Video processed successfully!",
        color: "green",
        icon: <IconCircleCheck />,
        autoClose: 5000,
      });
    }
  }, [success]);

  // Video synchronization effect
  useEffect(() => {
    // Only set up synchronization when both videos are available
    if (plotVideo && video) {
      // Wait for refs to be populated with actual video elements
      const checkForVideoElements = setInterval(() => {
        if (mainVideoRef.current && plotVideoRef.current) {
          clearInterval(checkForVideoElements);

          // Set up synchronization
          console.log("Setting up video synchronization");
          const cleanup = synchronizeVideos(
            mainVideoRef.current,
            plotVideoRef.current,
            { keepSecondaryMuted: true }
          );

          // Store cleanup function
          syncCleanupRef.current = cleanup;
        }
      }, 500);

      return () => {
        clearInterval(checkForVideoElements);
        // Clean up synchronization when component unmounts or videos change
        if (syncCleanupRef.current) {
          syncCleanupRef.current();
          syncCleanupRef.current = null;
        }
      };
    }
  }, [plotVideo, video]);

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

    // Clean up object URLs when component unmounts
    return () => {
      if (videoUrl) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [storedVideoName, storedPlotUrl]);

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

      // Create a URL for the video visualization (temporary, for the current session)
      const url = URL.createObjectURL(selectedVideo);
      setVideoUrl(url);

      // Store video in IndexedDB
      try {
        await storeVideo("current-video", selectedVideo, {
          name: selectedVideo.name,
          type: selectedVideo.type,
          size: selectedVideo.size,
        });

        // Store just the metadata in localStorage for quick checks
        setStoredVideoName(selectedVideo.name);

        // Clear stored plot URL since we have a new video
        setStoredPlotUrl(null);

        videoNameRef.current = selectedVideo.name;

        // Show notification for video upload
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

  const handleReset = () => {
    // Clear everything when resetting
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

  const handleVideoProcess = () => {
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
        } else {
          console.warn("Plot path not found in response");
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

  return (
    <Flex direction="column" gap={15}>
      {/* Main heading */}
      <HeadingLayout>
        <Title order={1}>Video Anomaly Detection</Title>
      </HeadingLayout>

      {/* Main content layout - With container to control overall positioning */}
      <Flex
        style={{
          paddingLeft: leftPadding,
          transition: `padding ${transitionDuration} ease`,
        }}
      >
        {/* Components layout */}
        <Flex
          justify="flex-start"
          align="start"
          gap="lg"
          wrap="nowrap"
          style={{
            transition: `all ${transitionDuration} ease`,
            width: "100%",
          }}
        >
          {/* Upload Video Section - Always aligned to left */}
          <Box
            style={{
              width: "49%",
              maxWidth: "49%",
              transition: `all ${transitionDuration} ease-in-out`,
            }}
          >
            <VideoUpload
              onVideoSelect={handleVideoSelect}
              height={componentHeight}
              width="100%"
              initialVideo={video}
              initialVideoUrl={videoUrl}
              videoRef={mainVideoRef}
            />
          </Box>

          {/* Anomaly Score Section - Space always reserved but content conditional */}
          <Box
            style={{
              width: "49%",
              maxWidth: "49%",
              transition: `all ${transitionDuration} ease-in-out`,
              visibility: video || loading ? "visible" : "hidden",
              opacity: video || loading ? 1 : 0,
            }}
          >
            <AnomalyPlot
              plotVideo={plotVideo}
              loading={loading}
              height={componentHeight}
              width="100%"
              videoRef={plotVideoRef}
            />
          </Box>
        </Flex>
      </Flex>

      {/* Process button */}
      {video && (
        <Flex justify="center" mt={rem(24)}>
          <Group>
            {!plotVideo && (
              <Button
                size="lg"
                radius="md"
                leftSection={<IconPlayerPlay size={20} />}
                onClick={handleVideoProcess}
                loading={processing}
                disabled={processing}
                style={{
                  ...processButtonStyle,
                  ...readyButtonStyle,
                }}
                styles={{
                  root: {
                    height: rem(50),
                    padding: `0 ${rem(35)}`,
                    fontSize: rem(16),
                    "&:hover": {
                      transform: "translateY(-2px)",
                      boxShadow: `0 ${rem(4)} ${rem(
                        12
                      )} rgba(34, 139, 230, 0.4)`,
                      background: "linear-gradient(45deg, #1c7ed6, #3dabf7)",
                    },
                    "&:active": {
                      transform: "translateY(0)",
                    },
                  },
                  inner: {
                    justifyContent: "center",
                  },
                }}
              >
                {processing ? "Processing..." : "Detect Anomalies"}
              </Button>
            )}

            {plotVideo && (
              <>
                <Button
                  size="lg"
                  radius="md"
                  variant="filled" // Thay vì disabled
                  leftSection={<IconCheck size={20} />}
                  style={{
                    ...processButtonStyle,
                    ...completeButtonStyle,
                    cursor: "default", // Đổi cursor thành default thay vì pointer
                    pointerEvents: "none", // Ngăn không cho click
                  }}
                  styles={{
                    root: {
                      height: rem(50),
                      padding: `0 ${rem(35)}`,
                      fontSize: rem(16),
                      "&:hover": {
                        background: "linear-gradient(45deg, #40c057, #82c91e)", // Giữ gradient không đổi khi hover
                      },
                      opacity: 0.85, // Tạo cảm giác disabled nhưng vẫn rõ màu
                    },
                  }}
                >
                  Analysis Complete
                </Button>

                <Button
                  size="lg"
                  radius="md"
                  variant="outline"
                  leftSection={<IconRefresh size={20} />}
                  onClick={handleReset}
                  styles={{
                    root: {
                      height: rem(50),
                    },
                  }}
                >
                  Run New Analysis
                </Button>
              </>
            )}
          </Group>
        </Flex>
      )}
    </Flex>
  );
}
