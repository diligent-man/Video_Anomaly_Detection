import { createContext, useState } from 'react';

export const NavbarContext = createContext({
  navbarOpened: true,
  setNavbarOpened: () => {}
});

export const NavbarProvider = ({ children }) => {
  const [navbarOpened, setNavbarOpened] = useState(true);

  return (
    <NavbarContext.Provider value={{ navbarOpened, setNavbarOpened }}>
      {children}
    </NavbarContext.Provider>
  );
};