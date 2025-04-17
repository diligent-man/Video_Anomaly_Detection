import { Title, Text, Button, Container, Stack, Group, Image, Box, useMantineColorScheme, Flex } from "@mantine/core";
import { IconArrowRight } from "@tabler/icons-react";
import { useNavigate } from "react-router-dom";
import landingIllustration from "../../assets/landing-about.jpg"; // You'll need to add this image
import styles from "./style.module.css";

export default function LandingPage() {
  const navigate = useNavigate();
  const { colorScheme } = useMantineColorScheme();
  const isDark = colorScheme === 'dark';

  const handleEnterDashboard = () => {
    navigate("/dashboard");
  };

  return (
    <div className={styles.landingPageWrapper}>
      <Container size="xl" className={styles.heroContainer}>
        <Flex 
          direction={{ base: 'column', md: 'row' }} 
          align="center" 
          gap={{ base: 'xl', md: '2rem' }} 
          justify="space-between"
          style={{ minHeight: '90vh' }}
        >
          {/* Left side content */}
          <Stack spacing="xl" style={{ maxWidth: 600 }}>
            <Title 
              className={styles.heroTitle}
              order={1} 
              style={{ fontSize: '3.2rem', lineHeight: 1.2 }}
            >
              Video Anomaly Detection
              <Box component="span" style={{ color: '#228be6', display: 'block' }}>
                Made Simple
              </Box>
            </Title>

            <Text size="xl" style={{ lineHeight: 1.6 }}>
              Detect unusual events in your videos with advanced AI technology. 
              Upload your footage and get insightful analysis in moments.
            </Text>

            <Group mt="xl">
              <Button 
                size="xl" 
                radius="md" 
                rightSection={<IconArrowRight size={20} />}
                onClick={handleEnterDashboard}
                className={styles.ctaButton}
              >
                Enter Dashboard
              </Button>
            </Group>

            <Text size="sm" color="dimmed" mt="md">
              Simple, powerful, and accurate video anomaly detection
            </Text>
          </Stack>

          {/* Right side illustration */}
          <Box
            className={styles.imageWrapper}
            style={{
              position: 'relative',
              maxWidth: '100%',
              width: 500,
              height: 400
            }}
          >
            <div className={styles.gradient}></div>
            <Image
              src={landingIllustration || "https://placehold.co/600x400?text=AnomaDetect"}
              alt="Video Anomaly Detection"
              className={styles.heroImage}
            />
          </Box>
        </Flex>

        {/* Features section */}
        <Box mt={{ base: 60, md: 100 }} className={styles.featuresSection}>
          <Flex 
            gap="xl" 
            justify="space-between" 
            align="flex-start"
            direction={{ base: 'column', md: 'row' }}
          >
            <div className={styles.featureItem}>
              <Title order={4} mb="md">Advanced Analysis</Title>
              <Text>Our AI technology identifies unusual patterns and behaviors in video content.</Text>
            </div>

            <div className={styles.featureItem}>
              <Title order={4} mb="md">Real-time Results</Title>
              <Text>Get detailed visual feedback on anomalies with synchronized playback.</Text>
            </div>

            <div className={styles.featureItem}>
              <Title order={4} mb="md">Secure Storage</Title>
              <Text>All your videos and analysis results are securely stored and easily accessible.</Text>
            </div>
          </Flex>
        </Box>
      </Container>
    </div>
  );
}