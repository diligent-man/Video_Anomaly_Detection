import React from "react";
import { Modal, Text, Group, Button } from "@mantine/core";

export function DeleteConfirmationModal({
  opened,
  onClose,
  onConfirm,
  videoName,
  loading,
}) {
  return (
    <Modal
      opened={opened}
      onClose={onClose}
      title={
        <Text fw={600} size="lg">Confirm Deletion</Text>
      }
      size="md"
      centered
    >
      <Text mb="lg">
        Are you sure you want to delete{" "}
        <strong>{videoName || "this video"}</strong>? This action cannot be
        undone.
      </Text>

      <Group justify="flex-end" mt="xl">
        <Button variant="outline" onClick={onClose} disabled={loading}>
          Cancel
        </Button>
        <Button color="red" onClick={onConfirm} loading={loading}>
          Delete
        </Button>
      </Group>
    </Modal>
  );
}