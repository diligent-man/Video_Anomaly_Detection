import DashboardPageLayout from "../pages/Dashboard/PageLayout";
import HomePage from "../pages/Dashboard/Home";
import LandingPage from "../pages/LandingPage/LandingPage";
import StoragePage from "../pages/Dashboard/Storage";
// import SettingPage from "../pages/Dashboard/Setting";

const appRoutes  = [
    {path: "/",
    element: <LandingPage/>,
    },
    {
        path: "/dashboard",
        element: <DashboardPageLayout/>,
        children: [
          {
            path: "/dashboard",
            element: <HomePage/>,
          },
          {
            path: "/dashboard/storage",
            element:<StoragePage/>,
          },


          // {
          //   path: "/dashboard/setting",
          //   element: <p>Setting </p>,
          // },

        ],
        
    },
      
]

export default appRoutes;