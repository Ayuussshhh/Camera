import { useEffect, useMemo, useState } from "react";
import { alpha, styled, useTheme } from "@mui/material/styles";
import {
  Box,
  Drawer,
  Paper,
  Toolbar,
  useMediaQuery
} from "@mui/material";
import themeConfig from "@/configs/themeConfig";
import SidebarContent from "@/layouts/components/vertical/SidebarContent";
import { MobileBottomNavigation } from "@/layouts/components/BottomNavigation";

const MainContentWrapper = styled(Box)(({ theme }) => ({
  flexGrow: 1,
  minWidth: 0,
  minHeight: "100vh",
  display: "flex",
  flexDirection: "column",
  background: `radial-gradient(circle at top right, ${alpha(
    theme.palette.primary.main,
    theme.palette.mode === "light" ? 0.08 : 0.14
  )} 0%, transparent 28%), ${theme.palette.background.default}`
}));

const ContentWrapper = styled("main", {
  shouldForwardProp: (prop) => prop !== "showBottomNav"
})(({ theme, showBottomNav }) => ({
  flexGrow: 1,
  width: "100%",
  padding: theme.spacing(2, 3, showBottomNav ? 13 : 4, 3),
  [theme.breakpoints.down("md")]: {
    padding: theme.spacing(2, 2, showBottomNav ? 12 : 4, 2)
  }
}));

export default function VerticalLayout({
  children,
  navigationItems = [],
  appBarContent,
  showBottomNav = true
}) {
  const NAV_COLLAPSE_STORAGE_KEY = "facetrace.navCollapsed";
  const theme = useTheme();
  const hidden = useMediaQuery(theme.breakpoints.down("lg"));
  const [navVisible, setNavVisible] = useState(false);
  const [navCollapsed, setNavCollapsed] = useState(false);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    setNavCollapsed(window.localStorage.getItem(NAV_COLLAPSE_STORAGE_KEY) === "true");
  }, []);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    window.localStorage.setItem(NAV_COLLAPSE_STORAGE_KEY, String(navCollapsed));
  }, [navCollapsed]);

  const desktopCollapsed = !hidden && navCollapsed;
  const drawerWidth = desktopCollapsed
    ? themeConfig.collapsedNavigationSize
    : themeConfig.navigationSize;
  const mobileNavItems = useMemo(
    () =>
      navigationItems
        .filter((item) => item.path)
        .slice(0, 4),
    [navigationItems]
  );

  const toggleNavVisibility = () => {
    setNavVisible((currentValue) => !currentValue);
  };

  const toggleDesktopNavCollapse = () => {
    setNavCollapsed((currentValue) => !currentValue);
  };

  const drawerContent = (
    <SidebarContent
      collapsed={desktopCollapsed}
      navigationItems={navigationItems}
      onNavigate={() => setNavVisible(false)}
    />
  );

  return (
    <Box sx={{ display: "flex", minHeight: "100vh" }}>
      <Box
        component="nav"
        sx={{
          width: { lg: drawerWidth },
          flexShrink: { lg: 0 },
          transition: theme.transitions.create("width", {
            duration: theme.transitions.duration.shorter,
            easing: theme.transitions.easing.sharp
          })
        }}
      >
        <Drawer
          ModalProps={{ keepMounted: true }}
          onClose={toggleNavVisibility}
          open={hidden ? navVisible : true}
          sx={{
            display: { xs: "block", lg: "block" },
            "& .MuiDrawer-paper": {
              width: drawerWidth,
              boxSizing: "border-box",
              overflowX: "hidden",
              borderRight: `1px solid ${theme.palette.divider}`,
              transition: theme.transitions.create(["width", "box-shadow"], {
                duration: theme.transitions.duration.shorter,
                easing: theme.transitions.easing.sharp
              })
            }
          }}
          variant={hidden ? "temporary" : "permanent"}
        >
          {drawerContent}
        </Drawer>
      </Box>

      <MainContentWrapper className="layout-content-wrapper">
        <Box
          component="header"
          sx={{
            position: "sticky",
            top: 0,
            zIndex: theme.zIndex.appBar,
            px: { xs: 2, md: 3 },
            pt: 2
          }}
        >
          <Paper
            elevation={0}
            sx={{
              border: `1px solid ${theme.palette.divider}`,
              backgroundColor: alpha(theme.palette.background.paper, themeConfig.appBarBlur ? 0.92 : 1),
              backdropFilter: themeConfig.appBarBlur ? "blur(14px)" : "none",
              boxShadow: "0px 2px 12px rgba(67, 89, 113, 0.08)",
              borderRadius: 2,
              overflow: "hidden"
            }}
          >
            <Toolbar
              disableGutters
              sx={{
                minHeight: "74px !important",
                px: { xs: "18px !important", md: "28px !important" }
              }}
            >
              {appBarContent?.({
                hidden,
                navCollapsed: desktopCollapsed,
                toggleDesktopNavCollapse,
                toggleNavVisibility
              })}
            </Toolbar>
          </Paper>
        </Box>

        <ContentWrapper showBottomNav={hidden && showBottomNav && mobileNavItems.length > 0}>
          {children}
        </ContentWrapper>
      </MainContentWrapper>

      {hidden && showBottomNav && mobileNavItems.length > 0 ? (
        <MobileBottomNavigation items={mobileNavItems} />
      ) : null}
    </Box>
  );
}
