import { useRef, useEffect, useState } from "react";
import {
  Paper,
  Title,
  Box,
  Center,
  Loader,
  Text,
  Stack,
  Alert,
  Skeleton,
} from "@mantine/core";
import { IconAlertCircle } from "@tabler/icons-react";

export default function AnomalyPlot({
  plotVideo,
  loading,
  height = 400,
  width = 500,
  videoRef = null // Accept external ref
}) {
  const [loadError, setLoadError] = useState(false);
  const internalVideoRef = useRef(null);
  
  // Use external ref if provided, otherwise use internal
  const actualVideoRef = videoRef || internalVideoRef;
  
  // Reset error state when plotVideo changes
  useEffect(() => {
    setLoadError(false);
  }, [plotVideo]);
  
  // Không cần event listeners nữa vì dùng synchronizeVideos
  
  const handleVideoError = () => {
    console.error("Error loading plot video");
    setLoadError(true);
  };
  
  // Ước tính chiều cao của Group trong VideoUpload (tên file + margin)
  const bottomSpacerHeight = "28px";

  // Xác định khi nào cần hiển thị spacer (khi có video hoặc placeholder)
  const showSpacer = true; // Luôn hiển thị spacer vì đã bỏ text

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
        borderWidth: "1px",
        borderStyle: "solid",
        borderColor: "#ced4da",
      }}
    >
      <Stack gap="sm" style={{ flex: 1, minHeight: 0 }}>
        <Title order={6}>Anomaly Score Visualization</Title>

        <Box
          style={{
            flex: 1,
            minHeight: 0,
            display: "flex",
            flexDirection: "column",
          }}
        >
          {loading ? (
            <Box
              style={{
                border: "2px dashed #ced4da",
                borderRadius: "18px",
                padding: "40px 20px",
                backgroundColor: "rgba(0, 0, 0, 0.03)",
                flex: 1,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                minHeight: 0,
              }}
            >
              <Center style={{ width: "100%", height: "100%" }}>
                <Loader size="lg" variant="bars" />
                <Text style={{ marginLeft: "16px" }}>Processing video...</Text>
              </Center>
            </Box>
          ) : plotVideo ? (
            loadError ? (
              <Alert
                icon={<IconAlertCircle size={16} />}
                title="Video Load Error"
                color="red"
              >
                Failed to load the anomaly score visualization. Please try
                refreshing the page.
              </Alert>
            ) : (
              <Box
                style={{
                  flex: 1,
                  display: "flex",
                  flexDirection: "column",
                  minHeight: 0,
                }}
              >
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
                    src={plotVideo}
                    style={{
                      width: "100%",
                      height: "100%",
                      objectFit: "contain",
                      display: "block",
                    }}
                    controls
                    playsInline
                    muted
                    onError={handleVideoError}
                    // Đã xóa tất cả các event handlers vì dùng SynchronizeVideo
                  />
                </Box>
              </Box>
            )
          ) : (
            <Box
              style={{
                border: "2px dashed #ced4da",
                borderRadius: "18px",
                padding: "40px 20px",
                textAlign: "center",
                backgroundColor: "rgba(0, 0, 0, 0.03)",
                flex: 1,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                minHeight: 0,
              }}
            >
              <Text c="dimmed">
                Press Detect Anomalies to see anomaly score visualization
              </Text>
            </Box>
          )}
        </Box>

        {/* Sử dụng spacer để giữ khoảng cách giống VideoUpload */}
        <Box style={{ height: bottomSpacerHeight, flexShrink: 0 }} />
      </Stack>
    </Paper>
  );
}