import { useMemo } from "react";
import { useRouter } from "next/router";
import {
  BottomNavigation,
  BottomNavigationAction,
  Paper
} from "@mui/material";
import { alpha, useTheme } from "@mui/material/styles";

function isActiveRoute(pathname, path) {
  if (path === "/dashboard") {
    return pathname === path;
  }

  return pathname === path || pathname.startsWith(`${path}/`);
}

export function MobileBottomNavigation({ items = [] }) {
  const router = useRouter();
  const theme = useTheme();

  const activeValue = useMemo(
    () =>
      Math.max(
        items.findIndex((item) => isActiveRoute(router.pathname, item.path)),
        0
      ),
    [items, router.pathname]
  );

  return (
    <Paper
      elevation={0}
      sx={{
        position: "fixed",
        left: 0,
        right: 0,
        bottom: 0,
        zIndex: theme.zIndex.appBar + 1,
        borderRadius: 0,
        borderTop: `1px solid ${theme.palette.divider}`,
        backgroundColor: alpha(theme.palette.background.paper, 0.96),
        backdropFilter: "blur(12px)"
      }}
    >
      <BottomNavigation
        onChange={(_, nextValue) => router.push(items[nextValue].path)}
        showLabels
        sx={{
          height: 76,
          "& .MuiBottomNavigationAction-root": {
            minWidth: 70,
            gap: 0.35,
            "&.Mui-selected": {
              color: "primary.main",
              "& .MuiBottomNavigationAction-label": {
                fontWeight: 700
              }
            }
          },
          "& .MuiBottomNavigationAction-label": {
            fontSize: "0.72rem"
          }
        }}
        value={activeValue}
      >
        {items.map((item) => {
          const Icon = item.icon;

          return (
            <BottomNavigationAction
              icon={<Icon fontSize="small" />}
              key={item.path}
              label={item.title}
            />
          );
        })}
      </BottomNavigation>
    </Paper>
  );
}
