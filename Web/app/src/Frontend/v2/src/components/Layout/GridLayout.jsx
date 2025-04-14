import { Flex, SimpleGrid, Title, Skeleton } from "@mantine/core";
import { Fragment } from "react";

export default function GridLayout({ title, children, loading = false }) {
  return (
    <Flex direction="column" gap="md">
      <Skeleton visible={loading} width="20%">
        {title ? <Title order={4}>{title}</Title> : null}
      </Skeleton>
      <SimpleGrid cols={{ base: 1, sm: 2, lg: 3, xl: 4 }}>
        {loading ? (
          <Fragment>
            <Skeleton height={90} />
            <Skeleton height={90} />
          </Fragment>
        ) : (
          children
        )}
      </SimpleGrid>
    </Flex>
  );
}