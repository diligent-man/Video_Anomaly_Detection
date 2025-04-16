import logo from "../../assets/logo512.png";

import { Flex, Title } from "@mantine/core";


export default function Logo({ size = 50, onTap, variant = "auto" }) {
  
 
  return (
    <Flex
      align="center"
      onClick={onTap}
      style={{
        cursor: "pointer",
      }}
    >
      <img
        src={logo}
        alt="logo"
        style={{ width: size, height: size }}
      />
      <Title
        ff="Roboto, sans-serif"
        order={3}
      >
        Video Anomaly Detection
      </Title>
    </Flex>
  );
}