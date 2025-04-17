import React, { useEffect, useRef, useState } from "react";
import { Modal, Group, Text, Flex, Box, Skeleton, Paper, Button } from "@mantine/core";
import { IconPlayerPlay } from '@tabler/icons-react';
import { synchronizeVideos } from "../../utils/SynchronizeVideo";
import styles from "./style.module.css";

export function VideoPreviewModal({
  opened,
  onClose,
  videoSrc,
  plotSrc,
  filename,
  loadingPlot,
  plotError,
}) {
  const originalVideoRef = useRef(null);
  const plotVideoRef = useRef(null);
  const syncRef = useRef(null);
  const [manualSync, setManualSync] = useState(false);

  // Cleanup function when component unmounts or modal closes
  useEffect(() => {
    return () => {
      if (syncRef.current) {
        syncRef.current();
        syncRef.current = null;
      }
    };
  }, []);

  // Reset videos when modal closes
  useEffect(() => {
    if (!opened) {
      if (originalVideoRef.current) {
        originalVideoRef.current.pause();
        originalVideoRef.current.currentTime = 0;
      }
      if (plotVideoRef.current) {
        plotVideoRef.current.pause();
        plotVideoRef.current.currentTime = 0;
      }
      if (syncRef.current) {
        syncRef.current();
        syncRef.current = null;
      }
    }
  }, [opened]);

  // Setup video synchronization when both videos are available
  useEffect(() => {
    if (!opened || !videoSrc || !plotSrc || plotError) {
      return;
    }

    // Wait for both videos to be ready
    const checkReadyAndSync = () => {
      if (!originalVideoRef.current || !plotVideoRef.current) return false;
      
      // Check if both videos have loaded enough data
      const originalReady = originalVideoRef.current.readyState >= 3; // HAVE_FUTURE_DATA
      const plotReady = plotVideoRef.current.readyState >= 3; // HAVE_FUTURE_DATA
      
      if (originalReady && plotReady) {
        console.log("Both videos are ready, setting up sync");
        
        // Clear any existing sync
        if (syncRef.current) {
          syncRef.current();
        }
        
        // Create new sync
        syncRef.current = synchronizeVideos(
          originalVideoRef.current,
          plotVideoRef.current,
          { keepSecondaryMuted: true }
        );
        
        return true;
      }
      return false;
    };
    
    // Try immediate sync
    if (!checkReadyAndSync()) {
      // If not ready, set up event listeners
      const readyHandler = () => {
        if (checkReadyAndSync()) {
          // Remove listeners once sync is established
          if (originalVideoRef.current) {
            originalVideoRef.current.removeEventListener('loadeddata', readyHandler);
            originalVideoRef.current.removeEventListener('canplay', readyHandler);
          }
          if (plotVideoRef.current) {
            plotVideoRef.current.removeEventListener('loadeddata', readyHandler);
            plotVideoRef.current.removeEventListener('canplay', readyHandler);
          }
        }
      };
      
      // Add event listeners for both videos
      if (originalVideoRef.current) {
        originalVideoRef.current.addEventListener('loadeddata', readyHandler);
        originalVideoRef.current.addEventListener('canplay', readyHandler);
      }
      if (plotVideoRef.current) {
        plotVideoRef.current.addEventListener('loadeddata', readyHandler);
        plotVideoRef.current.addEventListener('canplay', readyHandler);
      }
      
      // Safety timeout to ensure sync is attempted again
      const timeoutId = setTimeout(() => {
        checkReadyAndSync();
      }, 2000);
      
      return () => {
        clearTimeout(timeoutId);
        if (originalVideoRef.current) {
          originalVideoRef.current.removeEventListener('loadeddata', readyHandler);
          originalVideoRef.current.removeEventListener('canplay', readyHandler);
        }
        if (plotVideoRef.current) {
          plotVideoRef.current.removeEventListener('loadeddata', readyHandler);
          plotVideoRef.current.removeEventListener('canplay', readyHandler);
        }
      };
    }
  }, [opened, videoSrc, plotSrc, plotError, manualSync]);

  // Force manual sync function
  const forceSync = () => {
    if (syncRef.current) {
      syncRef.current();
    }
    
    if (originalVideoRef.current && plotVideoRef.current) {
      // Sync time
      plotVideoRef.current.currentTime = originalVideoRef.current.currentTime;
      
      // Sync play state
      if (!originalVideoRef.current.paused && plotVideoRef.current.paused) {
        plotVideoRef.current.play().catch(e => console.error("Error playing plot video:", e));
      } else if (originalVideoRef.current.paused && !plotVideoRef.current.paused) {
        plotVideoRef.current.pause();
      }
      
      // Setup sync again
      syncRef.current = synchronizeVideos(
        originalVideoRef.current,
        plotVideoRef.current,
        { keepSecondaryMuted: true }
      );
      
      setManualSync(prev => !prev); // Toggle to trigger useEffect
    }
  };
  
  // Render the plot content based on state
  const renderPlotContent = () => {
    if (loadingPlot) {
      return <Skeleton height="100%" minHeight={300} width="100%" radius="md" animate />;
    }

    if (plotError) {
      return (
        <Paper p="lg" withBorder className={styles.plotError}>
          <Text color="dimmed">{plotError}</Text>
        </Paper>
      );
    }

    if (plotSrc) {
      return (
        <Box className={styles.videoWrapper}>
          <video
            ref={plotVideoRef}
            key={plotSrc}
            src={plotSrc}
            controls
            muted
            playsInline
            className={styles.video}
            onLoadedMetadata={() => console.log("Plot video metadata loaded")}
          />
        </Box>
      );
    }

    // No plot available
    return (
      <Paper p="lg" withBorder className={styles.plotError}>
        <Text color="dimmed">Anomaly plot not available.</Text>
      </Paper>
    );
  };

  return (
    <Modal
      opened={opened}
      onClose={onClose}
      size="95%"
      title={
        <Group>
          <Text fw={700} size="lg">Video Preview</Text>
          {filename && <Text size="sm" color="dimmed">{filename}</Text>}
        </Group>
      }
    >
      <Flex
        gap="md"
        direction="column"
        align="stretch"
        className={styles.modalContent}
      >
        {/* Sync Controls */}
        {plotSrc && !plotError && (
          <Group position="center" mb="xs">
            {/* <Button 
              onClick={forceSync}
              leftSection={<IconPlayerPlay size={16} />}
              size="sm"
              variant="light"
            >
              Re-Sync Videos
            </Button> */}
          </Group>
        )}
        
        <Flex
          gap="md"
          direction={{ base: "column", md: "row" }}
          align="stretch"
          style={{ flexGrow: 1 }}
        >
          {/* Original Video */}
          <Box className={styles.videoContainer}>
            <Text fw={500} mb="xs">Video</Text>
            {videoSrc ? (
              <Box className={styles.videoWrapper}>
                <video
                  ref={originalVideoRef}
                  key={videoSrc}
                  src={videoSrc}
                  controls
                  playsInline
                  className={styles.video}
                  onLoadedMetadata={() => console.log("Original video metadata loaded")}
                />
              </Box>
            ) : (
              <Skeleton height={300} />
            )}
          </Box>

          {/* Anomaly Plot */}
          <Box className={styles.videoContainer}>
            <Text fw={500} mb="xs">Anomaly Plot</Text>
            {renderPlotContent()}
          </Box>
        </Flex>
      </Flex>
    </Modal>
  );
}