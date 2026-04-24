import { useEffect, useState } from "react";
import {
  Card,
  CardContent,
  FormControlLabel,
  Grid,
  MenuItem,
  Stack,
  Switch,
  TextField,
  Typography
} from "@mui/material";
import CustomDataGrid from "@/components/CustomDataGrid";
import { PrimaryButton } from "@/components/CustomButtons";
import settingsService from "@/services/settingsService";
import useAuth from "@/hooks/useAuth";
import { useAppSnackbar } from "@/context/SnackbarContext";
import { ENGINE_OPTIONS } from "@/utils/constants";
import { formatDateTime } from "@/utils/formatters";

export default function SettingsView() {
  const { user } = useAuth();
  const { showSnackbar } = useAppSnackbar();
  const [settings, setSettings] = useState(null);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    async function loadSettings() {
      try {
        const payload = await settingsService.getSettings();
        setSettings(payload);
      } catch (error) {
        showSnackbar(error.message || "Unable to load settings.", "error");
      }
    }

    loadSettings();
  }, [showSnackbar]);

  const handleChange = (key, value) => {
    setSettings((currentValue) => ({
      ...currentValue,
      [key]: value
    }));
  };

  const handleSave = async () => {
    try {
      setSaving(true);
      const payload = await settingsService.updateSettings({
        ...settings,
        actor: user
      });
      setSettings(payload);
      showSnackbar("Settings updated successfully.");
    } catch (error) {
      showSnackbar(error.message || "Unable to save settings.", "error");
    } finally {
      setSaving(false);
    }
  };

  if (!settings) {
    return null;
  }

  return (
    <Stack spacing={3}>
      <Grid container spacing={3}>
        <Grid item lg={8} xs={12}>
          <Card>
            <CardContent>
              <Stack spacing={2.5}>
                <Typography variant="h6">Recognition Configuration</Typography>
                <Grid container spacing={2}>
                  <Grid item md={6} xs={12}>
                    <TextField
                      fullWidth
                      label="Active Engine"
                      select
                      value={settings.activeEngine}
                      onChange={(event) => handleChange("activeEngine", event.target.value)}
                    >
                      {ENGINE_OPTIONS.map((engine) => (
                        <MenuItem key={engine.value} value={engine.value}>
                          {engine.label}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                  <Grid item md={6} xs={12}>
                    <TextField
                      fullWidth
                      label="Confidence Threshold"
                      type="number"
                      value={settings.confidenceThreshold}
                      onChange={(event) =>
                        handleChange("confidenceThreshold", Number(event.target.value))
                      }
                    />
                  </Grid>
                  <Grid item md={6} xs={12}>
                    <TextField
                      fullWidth
                      label="Duplicate Window (Minutes)"
                      type="number"
                      value={settings.duplicateWindowMinutes}
                      onChange={(event) =>
                        handleChange("duplicateWindowMinutes", Number(event.target.value))
                      }
                    />
                  </Grid>
                  <Grid item md={6} xs={12}>
                    <TextField
                      fullWidth
                      label="Auto Scan Interval (ms)"
                      type="number"
                      value={settings.autoScanIntervalMs}
                      onChange={(event) =>
                        handleChange("autoScanIntervalMs", Number(event.target.value))
                      }
                    />
                  </Grid>
                  <Grid item xs={12}>
                    <TextField
                      fullWidth
                      label="Default Camera"
                      value={settings.defaultCamera}
                      onChange={(event) => handleChange("defaultCamera", event.target.value)}
                    />
                  </Grid>
                </Grid>
                <Stack>
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.cameraEnabled}
                        onChange={(event) =>
                          handleChange("cameraEnabled", event.target.checked)
                        }
                      />
                    }
                    label="Camera Enabled"
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.livenessEnabled}
                        onChange={(event) =>
                          handleChange("livenessEnabled", event.target.checked)
                        }
                      />
                    }
                    label="Liveness / Anti-Spoof Ready"
                  />
                  <FormControlLabel
                    control={
                      <Switch
                        checked={settings.notificationsEnabled}
                        onChange={(event) =>
                          handleChange("notificationsEnabled", event.target.checked)
                        }
                      />
                    }
                    label="Notifications Enabled"
                  />
                </Stack>
                <PrimaryButton disabled={saving} onClick={handleSave}>
                  {saving ? "Saving..." : "Save Settings"}
                </PrimaryButton>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
        <Grid item lg={4} xs={12}>
          <Card>
            <CardContent>
              <Typography variant="h6">System Health</Typography>
              <Stack spacing={1.5} sx={{ mt: 2 }}>
                <Typography color="text.secondary" variant="body2">
                  Database Mode: {settings.health.databaseMode}
                </Typography>
                <Typography color="text.secondary" variant="body2">
                  Database Target: {settings.health.database || "-"}{settings.health.host ? ` @ ${settings.health.host}:${settings.health.port}` : ""}
                </Typography>
                <Typography color="text.secondary" variant="body2">
                  AI Service URL: {settings.health.aiServiceUrl}
                </Typography>
                <Typography color="text.secondary" variant="body2">
                  Enrolled Students: {settings.health.enrolledStudents} / {settings.health.totalStudents}
                </Typography>
                <Typography color="text.secondary" variant="body2">
                  Audit Entries: {settings.health.auditEntries}
                </Typography>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <CustomDataGrid
        columns={[
          { field: "actorName", headerName: "Actor", minWidth: 160, flex: 0.8 },
          { field: "action", headerName: "Action", minWidth: 150, flex: 0.7 },
          { field: "message", headerName: "Message", minWidth: 260, flex: 1.45 },
          {
            field: "createdAt",
            headerName: "Timestamp",
            minWidth: 190,
            flex: 0.9,
            render: (row) => formatDateTime(row.createdAt)
          }
        ]}
        rows={settings.auditLogs || []}
        title="Recent Audit Logs"
        subtitle="Every configuration save is preserved with actor and timestamp."
        showToolbar
      />
    </Stack>
  );
}
