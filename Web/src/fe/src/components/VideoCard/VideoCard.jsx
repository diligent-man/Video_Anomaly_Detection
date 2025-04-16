import React from "react";
import {
  Card,
  Image,
  Text,
  Group,
  Box,
  Skeleton,
  ActionIcon,
  Menu,
  Center,
} from "@mantine/core";
import { IconMovie, IconDots, IconTrash,IconPlayerPlay } from "@tabler/icons-react";
import styles from "./style.module.css";

// Helper function for date formatting
const formatDate = (dateString) => {
  if (!dateString) return "N/A";
  try {
    return new Date(dateString).toLocaleDateString();
  } catch (e) {
    return "Invalid Date";
  }
};

export function VideoCard({
  video,
  thumbnailUrl,
  isLoadingThumbnail,
  isDeleting,
  onCardClick,
  onDelete,
}) {
  // Track if thumbnail has failed to load
  const [thumbnailError, setThumbnailError] = React.useState(false);
  
  const handleCardClickInternal = (event) => {
    // Prevent triggering card click when clicking menu
    if (event.target.closest(`.${styles.menuContainer}`)) {
      event.stopPropagation();
      return;
    }
    onCardClick(video);
  };

  const handleDeleteClickInternal = (event) => {
    event.stopPropagation(); // Prevent card click
    onDelete(video);
  };
  
  // Handle thumbnail load error
  const handleThumbnailError = () => {
    console.log(`Thumbnail failed to load for: ${video.filename}`);
    setThumbnailError(true);
  };
  
  // Improved logic to determine if we should show the default icon
  const shouldShowPlaceholder = thumbnailError || 
    (!isLoadingThumbnail && 
      (!thumbnailUrl || 
       thumbnailUrl === "https://placehold.co/600x400/EEE/AAA?text=Loading..." ||
       thumbnailUrl === "data:," ||
       thumbnailUrl.includes("undefined")));

  return (
    <Card
      shadow="sm"
      padding="lg"
      radius="md"
      withBorder
      className={styles.videoCard}
      onClick={handleCardClickInternal}
      mih={240}
      style={{ display: "flex", flexDirection: "column", cursor: "pointer" }}
      data-testid={`video-card-${video.filename}`}
    >
      {/* Header with Title and Menu */}
      <Group justify="space-between" align="center" mb="md">
        <Text fw={500} truncate title={video.filename} style={{ flex: 1 }}>
          {video.filename}
        </Text>
        <Box className={styles.menuContainer}>
          <Menu position="bottom-end" shadow="md" withArrow withinPortal>
            <Menu.Target>
              <ActionIcon
                variant="subtle"
                color="gray"
                radius="sm"
                onClick={(e) => e.stopPropagation()}
                aria-label={`Actions for ${video.filename}`}
                disabled={isDeleting}
              >
                <IconDots size={16} />
              </ActionIcon>
            </Menu.Target>
            <Menu.Dropdown>
              <Menu.Item
                color="red"
                leftSection={<IconTrash size={16} />}
                onClick={handleDeleteClickInternal}
                disabled={isDeleting}
              >
                Delete video
              </Menu.Item>
            </Menu.Dropdown>
          </Menu>
        </Box>
      </Group>

      {/* Thumbnail Section */}
      <Card.Section style={{ flexGrow: 1, position: "relative" }}>
        {isLoadingThumbnail ? (
          <Skeleton visible height={160} width="100%" />
        ) : shouldShowPlaceholder ? (
          <Center 
            style={{ 
              height: 160, 
              backgroundColor: "#222", 
              width: "100%",
              display: "flex",
              justifyContent: "center",
              alignItems: "center"
            }}
          >
            <IconPlayerPlay
              size={64}
              color="#fff"
              style={{ filter: "drop-shadow(0 1px 3px rgba(0,0,0,0.8))" }}
            />
          </Center>
        ) : (
          <Image
            src={thumbnailUrl}
            height={160}
            alt={`Thumbnail for ${video.filename}`}
            fit="cover"
            onError={handleThumbnailError}
          />
        )}
        
        {/* Play icon overlay */}
        {!isLoadingThumbnail && !shouldShowPlaceholder && (
          <Box className={styles.playIcon}>
            <IconPlayerPlay
              size={36}
              color="#fff"
              style={{ filter: "drop-shadow(0 1px 2px rgba(0,0,0,0.5))" }}
            />
          </Box>
        )}
        
        {/* Spinner overlay for delete operation */}
        {isDeleting && (
          <Center 
            style={{ 
              position: 'absolute',
              top: 0,
              left: 0,
              right: 0,
              bottom: 0,
              backgroundColor: 'rgba(0,0,0,0.5)',
              zIndex: 10
            }}
          >
            <Text c="white" fw={600}>Deleting...</Text>
          </Center>
        )}
      </Card.Section>

      {/* Footer with Upload Date */}
      <Text size="sm" color="dimmed" mt="md">
        Uploaded: {formatDate(video.upload_date)}
      </Text>
    </Card>
  );
}