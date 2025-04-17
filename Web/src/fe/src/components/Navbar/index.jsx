import { useState, useEffect } from "react";
import { NavLink, Divider } from "@mantine/core";
import { useNavigate, useLocation } from "react-router-dom";

/**
 * Navbar items format
 * NavItem[] = [{
 *  label: string,
 *  description?: string,
 *  icon: React.ReactNode,
 *  activeIcon?: React.ReactNode,
 *  action: () => void,
 *  disabled?: boolean,
 *  rightSection?: React.ReactNode,
 *  path?: string, // Add this to track which route belongs to which item
 * }]
 */

export default function Navbar({
  items,
  activeIndex = 0,
  preItems = [],
  postItems = [],
}) {
  const navigate = useNavigate();
  const location = useLocation();
  
  // Initialize from localStorage or use default values
  const [navPreIndex, setNavPreIndex] = useState(() => {
    const saved = localStorage.getItem('navPreIndex');
    return saved !== null ? parseInt(saved) : null;
  });
  
  const [navIndex, setNavIndex] = useState(() => {
    const saved = localStorage.getItem('navIndex');
    return saved !== null ? parseInt(saved) : activeIndex;
  });
  
  const [navPostIndex, setNavPostIndex] = useState(() => {
    const saved = localStorage.getItem('navPostIndex');
    return saved !== null ? parseInt(saved) : null;
  });

  // Update localStorage when state changes
  useEffect(() => {
    if (navPreIndex !== null) localStorage.setItem('navPreIndex', navPreIndex);
    else localStorage.removeItem('navPreIndex');
  }, [navPreIndex]);

  useEffect(() => {
    if (navIndex !== null) localStorage.setItem('navIndex', navIndex);
    else localStorage.removeItem('navIndex');
  }, [navIndex]);

  useEffect(() => {
    if (navPostIndex !== null) localStorage.setItem('navPostIndex', navPostIndex);
    else localStorage.removeItem('navPostIndex');
  }, [navPostIndex]);

  // Optional: Sync navbar with current URL on initial load and route changes
  useEffect(() => {
    const currentPath = location.pathname;
    
    // Look for matching path in all item groups
    const findAndSetActiveItem = () => {
      // Check in main items
      const mainIndex = items.findIndex(item => item.path === currentPath);
      if (mainIndex >= 0) {
        setNavIndex(mainIndex);
        setNavPreIndex(null);
        setNavPostIndex(null);
        return true;
      }
      
      // Check in pre items
      const preIndex = preItems.findIndex(item => item.path === currentPath);
      if (preIndex >= 0) {
        setNavPreIndex(preIndex);
        setNavIndex(null);
        setNavPostIndex(null);
        return true;
      }
      
      // Check in post items
      const postIndex = postItems.findIndex(item => item.path === currentPath);
      if (postIndex >= 0) {
        setNavPostIndex(postIndex);
        setNavPreIndex(null);
        setNavIndex(null);
        return true;
      }
      
      return false;
    };
    
    findAndSetActiveItem();
  }, [location.pathname, items, preItems, postItems]);

  return (
    <>
      {preItems.map((item, index) => (
        <NavLink
          key={index}
          label={item.label}
          description={item.description}
          leftSection={
            navPreIndex === index ? item.activeIcon || item.icon : item.icon
          }
          active={navPreIndex === index}
          onClick={() => {
            setNavPreIndex(index);
            setNavPostIndex(null);
            setNavIndex(null);
            item.action();
            localStorage.setItem('navPreIndex', index);
            localStorage.removeItem('navIndex');
            localStorage.removeItem('navPostIndex');
          }}
          disabled={item.disabled}
        />
      ))}
      <Divider />
      {items.map((item, index) => (
        <NavLink
          key={index}
          label={item.label}
          description={item.description}
          leftSection={
            navIndex === index ? item.activeIcon || item.icon : item.icon
          }
          rightSection={item.rightSection || null}
          active={navIndex === index}
          onClick={() => {
            setNavIndex(index);
            setNavPostIndex(null);
            setNavPreIndex(null);
            item.action();
            localStorage.setItem('navIndex', index);
            localStorage.removeItem('navPreIndex');
            localStorage.removeItem('navPostIndex');
          }}
          disabled={item.disabled}
        />
      ))}
      {postItems.length > 0 && <Divider />}
      {postItems.map((item, index) => (
        <NavLink
          key={index}
          label={item.label}
          description={item.description}
          leftSection={
            navPostIndex === index ? item.activeIcon || item.icon : item.icon
          }
          active={navPostIndex === index}
          onClick={() => {
            setNavPostIndex(index);
            setNavPreIndex(null);
            setNavIndex(null);
            item.action();
            localStorage.setItem('navPostIndex', index);
            localStorage.removeItem('navPreIndex');
            localStorage.removeItem('navIndex');
          }}
          disabled={item.disabled}
        />
      ))}
    </>
  );
}