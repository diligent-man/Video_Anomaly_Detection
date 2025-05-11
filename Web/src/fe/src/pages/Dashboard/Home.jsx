import { Button, Flex, Title, Box, Group, rem } from "@mantine/core";
import { useRef, useContext, useEffect } from "react";
import {
  IconPlayerPlay,
  IconCheck,
  IconRefresh,
} from "@tabler/icons-react";
import HeadingLayout from "../../components/Layout/HeadingLayout.jsx";
import VideoUpload from "../../components/VideoUpload/VideoUpload.jsx";
import AnomalyPlot from "../../components/AnomalyPlot/AnomalyPlot.jsx";
import { NavbarContext } from "../../context/NavbarContext.jsx";
import { synchronizeVideos } from "../../utils/SynchronizeVideo";
import { VideoProcessingContext } from "../../context/VideoProcessingContext.jsx";

export default function HomePage() {
  // Get all state and functions from VideoProcessingContext
  const {
    video,
    videoUrl,
    processing,
    loading,
    plotVideo,
    success,
    handleVideoSelect,
    handleReset,
    processVideo
  } = useContext(VideoProcessingContext);

  // Get navbar state
  const { navbarOpened } = useContext(NavbarContext);

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
  const componentHeight = 500;
  const transitionDuration = "0.7s";

  // Add leftPadding adjustment based on navbar state
  const leftPadding = navbarOpened ? "0px" : "20px";

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
                onClick={processVideo}
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
                  variant="filled"
                  leftSection={<IconCheck size={20} />}
                  style={{
                    ...processButtonStyle,
                    ...completeButtonStyle,
                    cursor: "default",
                    pointerEvents: "none",
                  }}
                  styles={{
                    root: {
                      height: rem(50),
                      padding: `0 ${rem(35)}`,
                      fontSize: rem(16),
                      "&:hover": {
                        background: "linear-gradient(45deg, #40c057, #82c91e)",
                      },
                      opacity: 0.85,
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