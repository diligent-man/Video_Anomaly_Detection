import { Flex, Skeleton } from "@mantine/core";

export default function HeadingLayout({ children, loading = false }) {
  return (
    <Skeleton visible={loading}>
      <Flex
        align="flex-start"
        justify="space-between"
        style={{
          padding: "0 0 0 0",
        }}
      >
        {children}
      </Flex>
    </Skeleton>
  );
}