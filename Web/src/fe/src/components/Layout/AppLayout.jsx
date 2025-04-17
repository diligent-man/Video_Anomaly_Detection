import style from "./style.module.css";
import { AppShell, Flex, Container, ActionIcon } from "@mantine/core";
import { useDisclosure } from "@mantine/hooks";
import { useNavigate } from "react-router-dom";
import { useContext, useEffect } from "react";
import Logo from "../Logo";
import Navbar from "../Navbar";
import { IconChevronLeft, IconChevronRight } from '@tabler/icons-react';
import { NavbarContext } from "../../context/NavbarContext";
import { ThemeToggle } from "../ThemeToggle/ThemeToggle"; // Import ThemeToggle component

export default function AppLayout({
  children,
  navItems = [],
  navPreItems = [],
  navPostItems = [],
}) {
  const [opened, { toggle }] = useDisclosure(true);
  const { setNavbarOpened } = useContext(NavbarContext);
  const navigate = useNavigate();

  useEffect(() => {
    setNavbarOpened(opened);
  }, [opened, setNavbarOpened]);

  return (
    <AppShell
      navbar={{ 
        width: 250,
        breakpoint: "sm", 
        collapsed: { mobile: !opened, desktop: !opened }
      }}
      padding="md"
    >
      <AppShell.Navbar className={style.navbar}>
        <Flex justify="space-between" align="center" className={style.logoContainer}>
          <Logo size={30} />
          <ThemeToggle />
        </Flex>
        <Navbar preItems={navPreItems} items={navItems} postItems={navPostItems} />
        
        <div className={`${style.navbarToggle} ${opened ? style.navbarToggleOpen : style.navbarToggleCollapsed}`}>
          <ActionIcon 
            onClick={toggle}
            variant="outline"
            radius="xl"
            size="md"
          >
            {opened ? <IconChevronLeft size={18} /> : <IconChevronRight size={18} />}
          </ActionIcon>
        </div>
      </AppShell.Navbar>
      
      <AppShell.Main className={style.mainBackground}>
        <Container className={style.contentContainer} style={{ 
          transition: "all 0.4s ease",
          maxWidth: opened ? "100%" : "calc(100% + 220px)", 
          marginLeft: opened ? "0" : "-20px" 
        }}>
          {children}
        </Container>
      </AppShell.Main>
    </AppShell>
  );
}