import { useEffect, useState } from "react";
import {
  Box,
  Card,
  CardContent,
  Chip,
  Divider,
  Grid,
  List,
  ListItem,
  ListItemText,
  Stack,
  Typography
} from "@mui/material";
import DashboardCards from "@/components/DashboardCards";
import AttendanceStats from "@/components/AttendanceStats";
import CustomDataGrid from "@/components/CustomDataGrid";
import cameraService from "@/services/cameraService";
import reportService from "@/services/reportService";
import { useAppSnackbar } from "@/context/SnackbarContext";
import { getEngineLabel } from "@/utils/engine";
import { formatDateTime } from "@/utils/formatters";

export default function DashboardOverviewView() {
  const { showSnackbar } = useAppSnackbar();
  const [analytics, setAnalytics] = useState(null);
  const [cameraStatus, setCameraStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadDashboard() {
      try {
        const [analyticsPayload, cameraPayload] = await Promise.all([
          reportService.getAnalytics(),
          cameraService.getStatus()
        ]);

        setAnalytics(analyticsPayload);
        setCameraStatus(cameraPayload);
      } catch (error) {
        showSnackbar(error.message || "Unable to load dashboard.", "error");
      } finally {
        setLoading(false);
      }
    }

    loadDashboard();
  }, [showSnackbar]);

  return (
    <Stack spacing={3}>
      <Card>
        <CardContent>
          <Grid alignItems="center" container spacing={2.5}>
            <Grid item lg={8} xs={12}>
              <Typography variant="h5">College Operations Overview</Typography>
              <Typography color="text.secondary" sx={{ mt: 0.8 }} variant="body2">
                Centralized visibility across live recognition, attendance throughput, and camera
                system health.
              </Typography>
            </Grid>
            <Grid item lg={4} xs={12}>
              <Stack direction="row" flexWrap="wrap" gap={1} justifyContent={{ lg: "flex-end", xs: "flex-start" }}>
                <Chip
                  color={cameraStatus?.online ? "success" : "default"}
                  label={cameraStatus?.online ? "Camera Online" : "Camera Offline"}
                  size="small"
                />
                <Chip
                  color={cameraStatus?.aiHealth?.connected ? "success" : "warning"}
                  label={cameraStatus?.aiHealth?.connected ? "Python AI Live" : "Recognition Offline"}
                  size="small"
                />
                <Chip
                  color="primary"
                  label={`Sessions: ${(analytics?.sessions || []).length}`}
                  size="small"
                />
              </Stack>
            </Grid>
          </Grid>
        </CardContent>
      </Card>

      <DashboardCards cards={analytics?.cards || []} />
      <AttendanceStats activeSession={cameraStatus?.activeSession} summary={analytics?.summary} />

      <Grid container spacing={3}>
        <Grid item lg={8} xs={12}>
          <CustomDataGrid
            columns={[
              { field: "studentName", headerName: "Student", minWidth: 200, flex: 1.05 },
              { field: "rollNumber", headerName: "Roll Number", minWidth: 150, flex: 0.75 },
              { field: "subject", headerName: "Subject", minWidth: 160, flex: 0.85 },
              {
                field: "confidence",
                headerName: "Confidence",
                minWidth: 135,
                flex: 0.65,
                render: (row) => `${Math.round((row.confidence || 0) * 100)}%`
              },
              {
                field: "recognizedAt",
                headerName: "Recognized At",
                minWidth: 185,
                flex: 0.85,
                render: (row) => formatDateTime(row.recognizedAt)
              }
            ]}
            loading={loading}
            rows={analytics?.recentAttendance || []}
            showToolbar
            subtitle="Latest recognized students streaming in from live attendance sessions."
            title="Recent Attendance"
          />
        </Grid>
        <Grid item lg={4} xs={12}>
          <Stack spacing={3}>
            <Card>
              <CardContent>
                <Typography variant="h6">System Notifications</Typography>
                <List dense sx={{ mt: 1 }}>
                  {(analytics?.notifications || []).length ? (
                    (analytics?.notifications || []).map((notification) => (
                      <ListItem key={notification.id}>
                        <ListItemText
                          primary={
                            <Stack direction="row" justifyContent="space-between" spacing={1}>
                              <Typography variant="body2">{notification.title}</Typography>
                              <Chip color={notification.severity} label={notification.severity} size="small" />
                            </Stack>
                          }
                          secondary={`${notification.message} | ${formatDateTime(notification.createdAt)}`}
                        />
                      </ListItem>
                    ))
                  ) : (
                    <ListItem>
                      <ListItemText
                        primary="No active notifications"
                        secondary="New alerts, warnings, and system events will appear here."
                      />
                    </ListItem>
                  )}
                </List>
              </CardContent>
            </Card>

            <Card>
              <CardContent>
                <Typography variant="h6">Camera Status Panel</Typography>
                <Divider sx={{ my: 1.7 }} />
                <Stack spacing={1.1}>
                  {[
                    ["Device", cameraStatus?.deviceName || "-"],
                    ["Resolution", cameraStatus?.resolution || "-"],
                    ["FPS", cameraStatus?.fps || "-"],
                    ["Threshold", cameraStatus?.threshold || "-"],
                    ["Recognition Engine", cameraStatus?.engine ? getEngineLabel(cameraStatus.engine) : "-"],
                    ["AI Mode", cameraStatus?.aiHealth?.mode || "fallback"]
                  ].map(([label, value]) => (
                    <Box
                      key={label}
                      sx={{
                        display: "flex",
                        justifyContent: "space-between",
                        gap: 1,
                        py: 0.8
                      }}
                    >
                      <Typography color="text.secondary" variant="body2">
                        {label}
                      </Typography>
                      <Typography sx={{ fontWeight: 600 }} variant="body2">
                        {value}
                      </Typography>
                    </Box>
                  ))}
                </Stack>
              </CardContent>
            </Card>
          </Stack>
        </Grid>
        <Grid item xs={12}>
          <CustomDataGrid
            columns={[
              { field: "title", headerName: "Session", minWidth: 220, flex: 1.15 },
              { field: "departmentName", headerName: "Department", minWidth: 160, flex: 0.85 },
              { field: "subject", headerName: "Subject", minWidth: 160, flex: 0.85 },
              { field: "teacherName", headerName: "Teacher", minWidth: 160, flex: 0.85 },
              { field: "status", headerName: "Status", minWidth: 130, flex: 0.65 },
              {
                field: "startedAt",
                headerName: "Started At",
                minWidth: 185,
                flex: 0.85,
                render: (row) => formatDateTime(row.startedAt)
              }
            ]}
            loading={loading}
            rows={analytics?.sessions || []}
            showToolbar
            subtitle="A rolling window of live and completed classroom sessions."
            title="Recent Sessions"
          />
        </Grid>
      </Grid>
    </Stack>
  );
}
