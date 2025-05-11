import { useState, useEffect, useContext, useCallback, useRef } from "react";
import {
  Title,
  Grid,
  Text,
  Group,
  Box,
  Center,
  Flex,
  Paper,
  ActionIcon,
  Loader,
  Button,
  Badge,
} from "@mantine/core";
import {
  IconAlertCircle,
  IconRefresh,
  IconCheck,
  IconUpload,
} from "@tabler/icons-react";
import { notifications } from "@mantine/notifications";

// Layout Component
import HeadingLayout from "../../components/Layout/HeadingLayout";

// API and Utils
import { getAllVideosApi, getPlotUrl, checkScoreApi, deleteVideoApi } from "../../apis/Video";
import { baseUrl } from "../../utils/constants";
import { generateAndStoreThumbnail } from "../../utils/thumbnailUtils";
import { storeVideo, getVideo, clearVideo } from "../../utils/indexedDBStorage";

// Context
import { NavbarContext } from "../../context/NavbarContext";
import { VideoProcessingContext } from "../../context/VideoProcessingContext";

// Extracted Components
import { VideoCard } from "../../components/VideoCard/VideoCard";
import { VideoPreviewModal } from "../../components/VideoPreview/VideoPreview";
import { DeleteConfirmationModal } from "../../components/VideoPreview/DeleteModal";

// Import CSS Module
import classes from './style.module.css';

// Main Component
export default function StoragePage() {
  // Context hooks
  const { navbarOpened } = useContext(NavbarContext);
  const {
    video: currentProcessingVideo,
    processing,
    success,
    storedVideoName
  } = useContext(VideoProcessingContext);

  // State hooks
  const [videos, setVideos] = useState([]);
  const [loadingVideos, setLoadingVideos] = useState(true);
  const [fetchError, setFetchError] = useState(null);
  const [deletingVideos, setDeletingVideos] = useState({});
  const [showDeleteModal, setShowDeleteModal] = useState(false);
  const [videoToDelete, setVideoToDelete] = useState(null);
  const [showPreviewModal, setShowPreviewModal] = useState(false);
  const [previewVideoSrc, setPreviewVideoSrc] = useState(null);
  const [previewPlotSrc, setPreviewPlotSrc] = useState(null);
  const [previewFilename, setPreviewFilename] = useState(null);
  const [loadingPlot, setLoadingPlot] = useState(false);
  const [plotError, setPlotError] = useState(null);
  const [thumbnailUrls, setThumbnailUrls] = useState({});
  const [loadingThumbnails, setLoadingThumbnails] = useState({});

  // Track processing video
  const [processingVideoCard, setProcessingVideoCard] = useState(null);

  // Refs
  const thumbnailCleanupFunctionsRef = useRef([]);
  const processedVideosRef = useRef(new Set());
  const abortControllerRef = useRef(null);
  const processingSuccessRef = useRef(false);

  // Fetch videos directly instead of using the hook
  const fetchVideos = useCallback(async (forceRefresh = false) => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }

    abortControllerRef.current = new AbortController();

    try {
      setLoadingVideos(true);
      setFetchError(null);
      console.log("Fetching videos from API...");
      const fetchedVideos = await getAllVideosApi(abortControllerRef.current.signal);

      if (!fetchedVideos || !Array.isArray(fetchedVideos)) {
        console.error("Invalid response format:", fetchedVideos);
        setVideos([]);
        setFetchError("Received invalid data format from server");
      } else {
        console.log(`Fetched ${fetchedVideos.length} videos successfully`);
        setVideos(fetchedVideos);
        setFetchError(null);
      }
    } catch (err) {
      if (err.name !== 'AbortError') {
        console.error("Error fetching videos:", err);
        setFetchError(`Error loading videos: ${err.message}`);
      }
    } finally {
      setLoadingVideos(false);
    }
  }, []);

  // Initial fetch
  useEffect(() => {
    console.log("Initial fetch running");
    fetchVideos();

    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort();
      }
    };
  }, [fetchVideos]);

  // Create temporary processing video card when processing starts
  useEffect(() => {
    if (currentProcessingVideo && processing) {
      const videoName = currentProcessingVideo.name;
      setProcessingVideoCard({
        id: 'temp-processing-video',
        filename: videoName,
        isProcessing: true,
        createdAt: new Date().toISOString()
      });
      processingSuccessRef.current = false;
    }
  }, [currentProcessingVideo, processing]);

  // Listen for processing success to refresh list
  useEffect(() => {
    if (success && !processingSuccessRef.current && storedVideoName) {
      console.log('Processing completed successfully, refreshing videos');
      processingSuccessRef.current = true;

      // Wait a bit to ensure server has processed the video
      setTimeout(() => {
        fetchVideos(true);
        // Clear the processing card
        setProcessingVideoCard(null);
      }, 1500);
    }
  }, [success, fetchVideos, storedVideoName]);

  const handleOpenPreviewModal = useCallback(async (video) => {
    if (!video || !video.filename || video.isProcessing) return;

    setPreviewFilename(video.filename);

    try {
      setLoadingPlot(true);
      setPlotError(null);

      const videoUrl = `${baseUrl}apis/video/get_video/${encodeURIComponent(video.filename)}`;
      setPreviewVideoSrc(videoUrl);

      try {
        const plotUrl = await getPlotUrl(video.filename);
        setPreviewPlotSrc(plotUrl);
      } catch (err) {
        console.error("Error fetching plot:", err);
        setPlotError("Could not load the anomaly plot. The analysis may not have been completed.");
        setPreviewPlotSrc(null);
      } finally {
        setLoadingPlot(false);
      }

      setShowPreviewModal(true);

    } catch (err) {
      console.error("Error opening preview:", err);
      notifications.show({
        title: "Error",
        message: "Failed to open video preview",
        color: "red",
      });
    }
  }, [baseUrl]);

  const handleClosePreviewModal = useCallback(() => {
    setShowPreviewModal(false);
    setPreviewVideoSrc(null); // Clear the video source to stop playback
  }, []);

  const handleOpenDeleteModal = useCallback((video) => {
    if (!video || video.isProcessing) return;
    setVideoToDelete(video);
    setShowDeleteModal(true);
  }, []);

  const handleCloseDeleteModal = useCallback(() => {
    setShowDeleteModal(false);
    setVideoToDelete(null);
  }, []);

  const handleConfirmDelete = useCallback(async () => {
    if (!videoToDelete || !videoToDelete.filename) return;

    try {
      setDeletingVideos((prev) => ({
        ...prev,
        [videoToDelete.filename]: true,
      }));

      await deleteVideoApi(videoToDelete.filename);

      notifications.show({
        title: "Success",
        message: `Video ${videoToDelete.filename} has been deleted`,
        color: "green",
        icon: <IconCheck size={16} />,
      });

      try {
        await clearVideo(`thumbnail_${videoToDelete.filename}`);
      } catch (err) {
        console.error("Error clearing thumbnail from cache:", err);
      }

      setShowDeleteModal(false);
      setVideoToDelete(null);

      fetchVideos(true); // Explicitly force refresh after delete
    } catch (error) {
      console.error("Error deleting video:", error);
      notifications.show({
        title: "Error",
        message: `Failed to delete video: ${error.message}`,
        color: "red",
      });
    } finally {
      setDeletingVideos((prev) => ({
        ...prev,
        [videoToDelete?.filename]: false,
      }));
    }
  }, [videoToDelete, fetchVideos]);

  // Generate thumbnails in background
  useEffect(() => {
    if (!videos || videos.length === 0) return;

    const videosNeedingThumbnails = videos.filter(video =>
      video.filename &&
      !thumbnailUrls[video.filename] &&
      !processedVideosRef.current.has(video.filename)
    );

    if (videosNeedingThumbnails.length === 0) return;

    console.log(`Processing thumbnails for ${videosNeedingThumbnails.length} videos`);

    videosNeedingThumbnails.forEach(video => {
      processedVideosRef.current.add(video.filename);

      // Set initial loading state
      setLoadingThumbnails(prev => ({
        ...prev,
        [video.filename]: true
      }));

      // Try to get from cache first
      getVideo(`thumbnail_${video.filename}`)
        .then(savedThumbnail => {
          if (savedThumbnail?.metadata?.thumbnail) {
            console.log(`Using cached thumbnail for ${video.filename}`);
            setThumbnailUrls(prev => ({
              ...prev,
              [video.filename]: savedThumbnail.metadata.thumbnail
            }));
          } else {
            console.log(`Generating new thumbnail for ${video.filename}`);
            const cleanup = generateAndStoreThumbnail(
              video.filename,
              (thumbnailUrl) => {
                setThumbnailUrls(prev => ({
                  ...prev,
                  [video.filename]: thumbnailUrl
                }));
              },
              (error) => {
                console.error(`Error generating thumbnail: ${error}`);
              }
            );

            if (cleanup) thumbnailCleanupFunctionsRef.current.push(cleanup);
          }
        })
        .catch(error => {
          console.error(`IndexedDB error: ${error}`);
        })
        .finally(() => {
          setLoadingThumbnails(prev => ({
            ...prev,
            [video.filename]: false
          }));
        });
    });

    return () => {
      thumbnailCleanupFunctionsRef.current.forEach(cleanup => {
        if (typeof cleanup === 'function') {
          cleanup();
        }
      });
      thumbnailCleanupFunctionsRef.current = [];
    };
  }, [videos, thumbnailUrls]);

  // Reset tracking when videos are refetched
  useEffect(() => {
    if (loadingVideos) {
      processedVideosRef.current.clear();
    }
  }, [loadingVideos]);

  const transitionDuration = "0.7s";
  const leftPadding = navbarOpened ? "0px" : "20px";

  // Combine fetched videos with processing video if it exists
  const allVideos = processingVideoCard
    ? [processingVideoCard, ...videos]
    : videos;

  return (
    <Flex direction="column" gap="md" className={classes.container}>
      <HeadingLayout>
        <Group justify="space-between" w="100%">
          <Title order={1}>Video Storage</Title>
          <ActionIcon
            variant="light"
            color="blue"
            size="lg"
            onClick={() => fetchVideos(true)}
            disabled={loadingVideos}
            title="Refresh video list"
          >
            <IconRefresh size={20} />
          </ActionIcon>
        </Group>
      </HeadingLayout>

      {fetchError && (
        <Paper withBorder shadow="xs" p="md" m="md" className={classes.errorPaper}>
          <Group>
            <IconAlertCircle size={24} color="red" />
            <Text c="red">{fetchError}</Text>
            <Button ml="auto" onClick={() => fetchVideos(true)}>
              Retry
            </Button>
          </Group>
        </Paper>
      )}

      <Flex
        style={{
          paddingLeft: leftPadding,
          transition: `padding ${transitionDuration} ease`,
          width: "100%",
          flexGrow: 1,
        }}
      >
        <Box position="relative" style={{ width: '100%' }}>
          {allVideos.length === 0 && !fetchError && (
            <Center className={classes.centerMessage}>
              <Box style={{ textAlign: 'center' }}>
                {loadingVideos ? (
                  <>
                    <Loader size="md" mb="md" />
                    <Text size="lg" c="dimmed">Loading videos...</Text>
                  </>
                ) : (
                  <Text size="lg" c="dimmed">
                    No videos found. Upload videos via the Anomaly Detection tool.
                  </Text>
                )}
              </Box>
            </Center>
          )}

          {allVideos.length > 0 && (
            <Grid>
              {allVideos.map((video) => (
                <Grid.Col
                  span={{ base: 12, sm: 6, md: 4, lg: 3 }}
                  key={video.filename || video.id}
                >
                  {video.isProcessing ? (
                    <Paper shadow="md" p="md" radius="md" withBorder className={classes.processingCard}>
                      <Flex direction="column" align="center" justify="center" gap="md" style={{ height: '100%', minHeight: '200px' }}>
                        <Loader size="md" />
                        <Text fw={500} align="center">Processing video...</Text>
                        <Badge color="blue" leftSection={<IconUpload size={14} />}>
                          {video.filename}
                        </Badge>
                      </Flex>
                    </Paper>
                  ) : (
                    <VideoCard
                      video={video}
                      thumbnailUrl={thumbnailUrls[video.filename] || null}
                      isLoadingThumbnail={loadingThumbnails[video.filename] === true}
                      isDeleting={deletingVideos[video.filename] === true}
                      onCardClick={() => handleOpenPreviewModal(video)}
                      onDelete={() => handleOpenDeleteModal(video)}
                    />
                  )}
                </Grid.Col>
              ))}
            </Grid>
          )}
          
          {loadingVideos && videos.length > 0 && (
            <Box mt="md">
              <Center>
                <Group>
                  <Loader size="sm" />
                  <Text size="sm" c="dimmed">Refreshing video list...</Text>
                </Group>
              </Center>
            </Box>
          )}
        </Box>
      </Flex>

      <VideoPreviewModal
        opened={showPreviewModal}
        onClose={handleClosePreviewModal}
        videoSrc={previewVideoSrc}
        plotSrc={previewPlotSrc}
        filename={previewFilename}
        loadingPlot={loadingPlot}
        plotError={plotError}
      />

      <DeleteConfirmationModal
        opened={showDeleteModal}
        onClose={handleCloseDeleteModal}
        onConfirm={handleConfirmDelete}
        videoName={videoToDelete?.filename}
        loading={deletingVideos[videoToDelete?.filename] === true}
      />
    </Flex>
  );
}