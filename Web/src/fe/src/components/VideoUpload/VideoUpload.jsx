import { useState, useRef, useEffect } from "react";
import {
  Box,
  Text,
  Group,
  Paper,
  FileButton,
  Stack,
  Flex,
  Title,
  ActionIcon,
  Alert,
} from "@mantine/core";
import { IconUpload, IconX, IconAlertCircle } from "@tabler/icons-react";

const MAX_VIDEO_SIZE = 100 * 1024 * 1024; // 100MB max size

export default function VideoUpload({
  onVideoSelect,
  title = "Upload Video",
  height = 400,
  width = 500,
  initialVideo = null,
  initialVideoUrl = null,
  videoRef = null,
}) {
  const [file, setFile] = useState(initialVideo);
  const [videoSrc, setVideoSrc] = useState(initialVideoUrl);
  const [fileError, setFileError] = useState(null);
  const internalVideoRef = useRef(null);
  
  // Use external ref if provided, otherwise use internal
  const actualVideoRef = videoRef || internalVideoRef;

  // Handle initial video
  useEffect(() => {
    if (initialVideo && !videoSrc && initialVideoUrl) {
      setFile(initialVideo);
      setVideoSrc(initialVideoUrl);
    }
  }, [initialVideo, initialVideoUrl]);

  const handleFileChange = (selectedFile) => {
    if (!selectedFile) return;
    clearVideo(); // Xóa video cũ trước khi tải video mới
    setFileError(null);

    if (!selectedFile.type.startsWith("video/")) {
      setFileError("Please upload a valid video file");
      return;
    }

    if (selectedFile.size > MAX_VIDEO_SIZE) {
      setFileError(
        `Video file is too large (${(selectedFile.size / (1024 * 1024)).toFixed(
          2
        )}MB). Maximum size is ${MAX_VIDEO_SIZE / (1024 * 1024)}MB.`
      );
      return;
    }

    setFile(selectedFile);
    const videoUrl = URL.createObjectURL(selectedFile);
    setVideoSrc(videoUrl);

    if (onVideoSelect) {
      onVideoSelect(selectedFile);
    }
  };

  const clearVideo = () => {
    if (videoSrc && actualVideoRef.current) {
      // Dừng video và giải phóng URL
      actualVideoRef.current.pause();
      actualVideoRef.current.removeAttribute("src");
      actualVideoRef.current.load();
      
      // Don't revoke URL if it's the initialVideoUrl from localStorage
      if (videoSrc !== initialVideoUrl) {
        URL.revokeObjectURL(videoSrc);
      }
    }
    setFile(null);
    setVideoSrc(null);
    setFileError(null);
    if (onVideoSelect) {
      onVideoSelect(null);
    }
  };

  // Handle timeupdate event to keep videos in sync during playback
  const handleTimeUpdate = () => {
    if (actualVideoRef.current && actualVideoRef.current.paused === false) {
      // Only dispatch time update events periodically to avoid performance issues
      const now = Date.now();
      if (
        !handleTimeUpdate.lastUpdate ||
        now - handleTimeUpdate.lastUpdate > 1000
      ) {
        handleTimeUpdate.lastUpdate = now;
        const event = new CustomEvent("uploadVideoTimeUpdate", {
          detail: { time: actualVideoRef.current.currentTime },
        });
        document.dispatchEvent(event);
      }
    }
  };

  // Handle play, pause, and seeking events - Only send events, don't listen for them
  const handlePlay = () => {
    const event = new CustomEvent("uploadVideoPlay", {
      detail: { time: actualVideoRef.current.currentTime },
    });
    document.dispatchEvent(event);
  };

  const handlePause = () => {
    const event = new CustomEvent("uploadVideoPause");
    document.dispatchEvent(event);
  };

  const handleSeek = () => {
    const event = new CustomEvent("uploadVideoSeek", {
      detail: { time: actualVideoRef.current.currentTime },
    });
    document.dispatchEvent(event);
  };

  // Xác định khi nào hiển thị phần video (và group tên file)
  const showVideoContent = !fileError && videoSrc && file;

  return (
    <Paper
      p="md"
      radius="md"
      withBorder
      shadow="md"
      style={{
        height,
        width,
        display: "flex",
        flexDirection: "column",
        borderWidth: "1px", // Increase border thickness
        borderStyle: "solid",
        borderColor: "#ced4da",
      }}
    >
      <Stack gap="sm" style={{ flex: 1, minHeight: 0 }}>
        <Title order={6}>{title}</Title>

        {fileError && (
          <Alert
            icon={<IconAlertCircle size={16} />}
            title="Error"
            color="red"
            withCloseButton
            onClose={() => setFileError(null)}
          >
            {fileError}
          </Alert>
        )}

        <Box
          style={{
            flex: 1,
            minHeight: 0,
            display: "flex",
            flexDirection: "column",
          }}
        >
          {!showVideoContent ? (
            <Box
              style={{
                border: "2px dashed #ced4da",
                borderRadius: "18px",
                padding: "40px 20px",
                textAlign: "center",
                backgroundColor: "rgba(0, 0, 0, 0.03)",
                cursor: "pointer",
                flex: 1,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                minHeight: 0,
              }}
            >
              <FileButton onChange={handleFileChange} accept="video/*">
                {(props) => (
                  <Flex direction="column" align="center" gap="md" {...props}>
                    <IconUpload size={36} stroke={1.5} />
                    <Text size="xl" fw={500}>
                      Drag video here or click to upload
                    </Text>
                    <Text size="sm" c="dimmed">
                      Supports MP4, WebM, AVI, MOV and other video formats (max{" "}
                      {MAX_VIDEO_SIZE / (1024 * 1024)}MB)
                    </Text>
                  </Flex>
                )}
              </FileButton>
            </Box>
          ) : (
            <Box
              pos="relative"
              style={{
                flex: 1,
                display: "flex",
                flexDirection: "column",
                minHeight: 0,
              }}
            >
              <ActionIcon
                variant="filled"
                color="gray"
                radius="xl"
                size="md"
                onClick={clearVideo}
                style={{
                  position: "absolute",
                  top: "0",
                  right: "0",
                  zIndex: 10,
                  opacity: 0.8,
                  backgroundColor: "rgba(0, 0, 0, 0.6)",
                  color: "white",
                  borderRadius: "0 8px 0 8px",
                  transform: "translate(0, 0)",
                }}
                aria-label="Clear video"
              >
                <IconX size={16} />
              </ActionIcon>
              <Box
                style={{
                  flex: 1,
                  position: "relative",
                  overflow: "hidden",
                  borderRadius: "8px",
                  border: "1px solid #e0e0e0",
                  minHeight: 0,
                }}
              >
                <video
                  ref={actualVideoRef}
                  src={videoSrc}
                  style={{
                    width: "100%",
                    height: "100%",
                    objectFit: "contain",
                    display: "block",
                  }}
                  controls
                  playsInline
                  onPlay={handlePlay}
                  onPause={handlePause}
                  onSeeking={handleSeek}
                  onTimeUpdate={handleTimeUpdate}
                />
              </Box>
            </Box>
          )}
        </Box>

        {showVideoContent && (
          <Group mt="xs" justify="space-between" style={{ flexShrink: 0 }}>
            <Text size="sm" fw={500} truncate>
              {file.name} ({(file.size / (1024 * 1024)).toFixed(2)} MB)
            </Text>
          </Group>
        )}
      </Stack>
    </Paper>
  );
}