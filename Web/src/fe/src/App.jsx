import { MantineProvider } from "@mantine/core";
import { ModalsProvider } from "@mantine/modals";
import { useRoutes } from "react-router-dom";
import { useState, useEffect } from "react";
import appRoutes from "./routes/routes";
import { NavbarProvider } from "./context/NavbarContext"
import { useLocalStorage } from '@mantine/hooks';
import { Notifications } from "@mantine/notifications";

export default function App() {
  // Use useLocalStorage hook to persist the theme
  const [colorScheme, setColorScheme] = useLocalStorage({
    key: 'color-scheme',
    defaultValue: 'dark',
    getInitialValueInEffect: true,
  });

  // Theme toggle function
  const toggleColorScheme = (value) => {
    setColorScheme(value || (colorScheme === 'dark' ? 'light' : 'dark'));
  };

  // Check system preference initially if no preference is saved
  useEffect(() => {
    if (!localStorage.getItem('color-scheme')) {
      const prefersDarkMode = window.matchMedia('(prefers-color-scheme: dark)').matches;
      setColorScheme(prefersDarkMode ? 'dark' : 'light');
    }
  }, []);

  return (
    <MantineProvider
      theme={{
        colorScheme,
      }}
      defaultColorScheme={colorScheme}
    >
      <Notifications />
      <ModalsProvider>
        <NavbarProvider>
          {useRoutes(appRoutes)}
        </NavbarProvider>
      </ModalsProvider>
    </MantineProvider>
  );
}