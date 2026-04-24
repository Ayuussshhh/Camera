import {
  Avatar,
  Box,
  Badge,
  IconButton,
  Stack,
  Tooltip,
  Typography
} from "@mui/material";
import MenuRoundedIcon from "@mui/icons-material/MenuRounded";
import MenuOpenRoundedIcon from "@mui/icons-material/MenuOpenRounded";
import DarkModeRoundedIcon from "@mui/icons-material/DarkModeRounded";
import LightModeRoundedIcon from "@mui/icons-material/LightModeRounded";
import NotificationsNoneRoundedIcon from "@mui/icons-material/NotificationsNoneRounded";
import useAuth from "@/hooks/useAuth";
import { useThemeMode } from "@/context/ThemeModeContext";
import themeConfig from "@/configs/themeConfig";

export default function AppBarContent({
  hidden,
  navCollapsed,
  toggleDesktopNavCollapse,
  toggleNavVisibility,
  title,
  subtitle,
  actions
}) {
  const { user, logout } = useAuth();
  const { mode, toggleColorMode } = useThemeMode();
  const handleNavToggle = hidden ? toggleNavVisibility : toggleDesktopNavCollapse;
  const navToggleLabel = hidden
    ? "Open navigation"
    : navCollapsed
      ? "Expand sidebar"
      : "Collapse sidebar";

  return (
    <Box
      sx={{
        width: "100%",
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 2
      }}
    >
      <Stack direction="row" spacing={1.75} sx={{ alignItems: "center", minWidth: 0 }}>
        <Tooltip title={navToggleLabel}>
          <IconButton color="inherit" onClick={handleNavToggle} size="small">
            {hidden || navCollapsed ? <MenuRoundedIcon /> : <MenuOpenRoundedIcon />}
          </IconButton>
        </Tooltip>

        <Box
          sx={{
            minWidth: 0,
            display: "flex",
            flexDirection: "column"
          }}
        >
          <Typography
            color="text.secondary"
            noWrap
            sx={{ fontSize: "0.75rem", fontWeight: 700, letterSpacing: 0.9, textTransform: "uppercase" }}
            variant="caption"
          >
            {themeConfig.templateName}
          </Typography>
          <Typography noWrap sx={{ fontWeight: 700, lineHeight: 1.15, mt: 0.25 }} variant="h6">
            {title || "Dashboard Workspace"}
          </Typography>
          <Typography color="text.secondary" noWrap sx={{ mt: 0.3 }} variant="body2">
            {subtitle || "Engineering College"}
          </Typography>
        </Box>
      </Stack>

      <Stack direction="row" spacing={1.25} sx={{ alignItems: "center", flexShrink: 0 }}>
        {actions}

        <Tooltip title={mode === "light" ? "Enable dark mode" : "Enable light mode"}>
          <IconButton color="inherit" onClick={toggleColorMode} size="small">
            {mode === "light" ? <DarkModeRoundedIcon /> : <LightModeRoundedIcon />}
          </IconButton>
        </Tooltip>

        <Tooltip title="Notifications">
          <IconButton color="inherit" size="small">
            <NotificationsNoneRoundedIcon />
          </IconButton>
        </Tooltip>

        <Stack direction="row" spacing={1.25} sx={{ alignItems: "center" }}>
          <Tooltip title={`Logged in as ${user?.name || "Authenticated User"}. Click to logout.`}>
            <Badge
              anchorOrigin={{ horizontal: "right", vertical: "bottom" }}
              color="success"
              overlap="circular"
              variant="dot"
            >
              <Avatar
                onClick={logout}
                sx={{ bgcolor: "primary.main", width: 46, height: 46, cursor: "pointer" }}
              >
                {user?.name?.charAt(0)?.toUpperCase() || "U"}
              </Avatar>
            </Badge>
          </Tooltip>
        </Stack>
      </Stack>
    </Box>
  );
}
