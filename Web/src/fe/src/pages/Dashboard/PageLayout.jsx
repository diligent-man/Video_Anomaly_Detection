import { Outlet, useNavigate } from "react-router-dom";
import AppLayout from "../../components/Layout/AppLayout";
import { IconHome, IconHomeFilled, IconSettings, IconSettingsFilled,IconDatabase  } from "@tabler/icons-react";

export default function DashboardPageLayout() {
  const navigate = useNavigate();

  const navbarItems = [
    {
      label: "Home",
      icon: <IconHome size="1rem" />,
      activeIcon: <IconHomeFilled size="1rem" />,
      action: () => navigate("/dashboard"),
    },
    {
      label: "Storage",
      icon: <IconDatabase size="1rem" />,
      activeIcon: <IconDatabase size="1rem"  />,
      action: () => navigate("/dashboard/Storage"),
    },
  ];

  // const navbarSettings = [
  //   {
  //     label: "Settings",
  //     icon: <IconSettings size="1rem" />,
  //     activeIcon: <IconSettingsFilled size="1rem" />,
  //     action: () => navigate("/dashboard/setting"),
  //   },
  // ];

  return (
    <AppLayout navItems={navbarItems} >
      <Outlet />
    </AppLayout>
  );
}
// navPostItems={navbarSettings}