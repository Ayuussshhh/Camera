import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/router";
import {
  Badge,
  Box,
  List,
  ListItemButton,
  ListItemIcon,
  ListItemText,
  Tooltip,
  Typography
} from "@mui/material";
import { alpha, useTheme } from "@mui/material/styles";

function isActiveRoute(pathname, path) {
  if (path === "/dashboard") {
    return pathname === path;
  }

  return pathname === path || pathname.startsWith(`${path}/`);
}

export default function SidebarContent({
  collapsed = false,
  navigationItems = [],
  onNavigate
}) {
  const router = useRouter();
  const theme = useTheme();
  const activeGradient = `linear-gradient(118deg, ${theme.palette.primary.main} 0%, ${alpha(
    theme.palette.primary.main,
    0.76
  )} 100%)`;

  return (
    <Box
      sx={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        px: collapsed ? 1.1 : 1.75,
        py: 1.25
      }}
    >
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: collapsed ? "center" : "flex-start",
          minHeight: collapsed ? 92 : 112,
          px: collapsed ? 0 : 1.5,
          pb: 1.5
        }}
      >
        {collapsed ? (
          <Box
            sx={{
              width: 58,
              height: 58,
              borderRadius: 2,
              display: "grid",
              placeItems: "center",
              border: `1px solid ${alpha(theme.palette.primary.main, 0.18)}`,
              bgcolor: alpha(theme.palette.primary.main, 0.08)
            }}
          >
            <Image
              alt="CDGI College Logo"
              height={38}
              priority
              src="/images/pages/collegeLogo.png"
              width={38}
            />
          </Box>
        ) : (
          <Image
            alt="CDGI Sidebar Logo"
            height={78}
            priority
            src="/images/pages/LogoSidebar.png"
            width={230}
          />
        )}
      </Box>

      <List sx={{ mt: 0.75, px: collapsed ? 0 : 0.25 }}>
        {navigationItems.map((item) => {
          if (item.sectionTitle) {
            if (collapsed) {
              return null;
            }

            return (
              <Typography
                color="text.secondary"
                key={item.sectionTitle}
                sx={{
                  px: 2,
                  pt: 2.25,
                  pb: 0.8,
                  fontSize: "0.75rem",
                  fontWeight: 500,
                  letterSpacing: 0.8,
                  textTransform: "uppercase"
                }}
                variant="caption"
              >
                {item.sectionTitle}
              </Typography>
            );
          }

          const Icon = item.icon;
          const selected = isActiveRoute(router.pathname, item.path);
          const listItemButton = (
            <ListItemButton
              onClick={onNavigate}
              selected={selected}
              sx={{
                borderRadius: 2,
                py: collapsed ? 1.2 : 1.45,
                px: collapsed ? 1.1 : 2,
                mb: 0.7,
                minHeight: 48,
                justifyContent: collapsed ? "center" : "flex-start",
                alignItems: "center",
                transition: "all 0.2s ease",
                "&:hover": {
                  backgroundColor: alpha(theme.palette.primary.main, 0.08)
                },
                ...(selected && {
                  boxShadow: `0px 6px 14px ${alpha(theme.palette.primary.main, 0.22)}`,
                  background: activeGradient,
                  "&:hover": {
                    background: activeGradient
                  },
                  "& .MuiListItemText-primary, & .MuiListItemText-secondary, & .MuiSvgIcon-root": {
                    color: `${theme.palette.common.white} !important`
                  }
                })
              }}
            >
              <ListItemIcon
                sx={{
                  minWidth: collapsed ? 0 : 38,
                  mr: collapsed ? 0 : 0.5,
                  justifyContent: "center",
                  color: selected ? "common.white" : "text.secondary"
                }}
              >
                {item.badgeContent ? (
                  <Badge
                    badgeContent={collapsed ? null : item.badgeContent}
                    color={item.badgeColor || "secondary"}
                    sx={{ "& .MuiBadge-badge": { fontSize: 9, minWidth: 16, height: 16 } }}
                  >
                    <Icon fontSize="small" />
                  </Badge>
                ) : (
                  <Icon fontSize="small" />
                )}
              </ListItemIcon>
              {collapsed ? null : (
                <ListItemText
                  primary={item.title}
                  primaryTypographyProps={{ fontWeight: 600, variant: "body1" }}
                  secondary={item.caption}
                  secondaryTypographyProps={{
                    sx: { lineHeight: 1.35, mt: 0.2 },
                    variant: "caption"
                  }}
                />
              )}
            </ListItemButton>
          );

          return (
            <Tooltip
              arrow
              disableHoverListener={!collapsed}
              key={item.path}
              placement="right"
              title={item.title}
            >
              <Link href={item.path} style={{ textDecoration: "none" }}>
                {listItemButton}
              </Link>
            </Tooltip>
          );
        })}
      </List>
    </Box>
  );
}
