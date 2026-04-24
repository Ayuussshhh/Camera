import { useEffect, useRef, useState } from "react";
import {
  Alert,
  Box,
  Card,
  CardContent,
  Chip,
  Divider,
  Grid,
  List,
  ListItem,
  ListItemText,
  MenuItem,
  Stack,
  Switch,
  TextField,
  Typography
} from "@mui/material";
import BoltRoundedIcon from "@mui/icons-material/BoltRounded";
import PeopleAltRoundedIcon from "@mui/icons-material/PeopleAltRounded";
import PlayCircleRoundedIcon from "@mui/icons-material/PlayCircleRounded";
import RadarRoundedIcon from "@mui/icons-material/RadarRounded";
import StopCircleRoundedIcon from "@mui/icons-material/StopCircleRounded";
import { PrimaryButton, SecondaryButton } from "@/components/CustomButtons";
import { ENGINE_OPTIONS, SECTION_OPTIONS, SEMESTER_OPTIONS } from "@/utils/constants";
import { getEngineLabel } from "@/utils/engine";
import { formatDateTime } from "@/utils/formatters";

function getRecognitionTone(result) {
  if (result.attendanceStatus === "marked") {
    return "success";
  }

  if (result.attendanceStatus === "duplicate") {
    return "info";
  }

  if (result.attendanceStatus === "skipped") {
    return "warning";
  }

  return "default";
}

export default function CameraCard({
  cameraStatus,
  departments = [],
  eligibleStudents = [],
  loading,
  recognitions = [],
  rosterCount,
  sessionForm,
  manualState,
  onManualStateChange,
  onManualMark,
  onSessionFormChange,
  onStartSession,
  onStopSession,
  onRecognizeFrame
}) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const [stream, setStream] = useState(null);
  const [autoScan, setAutoScan] = useState(false);
  const activeSession = cameraStatus?.activeSession;

  useEffect(() => {
    return () => {
      if (stream) {
        stream.getTracks().forEach((track) => track.stop());
      }
    };
  }, [stream]);

  useEffect(() => {
    if (!activeSession || stream) {
      return;
    }

    async function startBrowserCamera() {
      if (!navigator.mediaDevices?.getUserMedia) {
        return;
      }

      const mediaStream = await navigator.mediaDevices.getUserMedia({ video: true });
      setStream(mediaStream);

      if (videoRef.current) {
        videoRef.current.srcObject = mediaStream;
        await videoRef.current.play();
      }
    }

    startBrowserCamera();
  }, [activeSession, stream]);

  useEffect(() => {
    if (!autoScan || !activeSession) {
      return undefined;
    }

    const timer = setInterval(() => {
      if (!videoRef.current || !canvasRef.current) {
        return;
      }

      const canvas = canvasRef.current;
      const video = videoRef.current;
      canvas.width = video.videoWidth || 640;
      canvas.height = video.videoHeight || 480;
      canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
      onRecognizeFrame(canvas.toDataURL("image/jpeg", 0.82));
    }, cameraStatus?.autoScanIntervalMs || 4000);

    return () => clearInterval(timer);
  }, [activeSession, autoScan, cameraStatus?.autoScanIntervalMs, onRecognizeFrame]);

  const scanFrame = () => {
    if (!videoRef.current || !canvasRef.current) {
      return;
    }

    const canvas = canvasRef.current;
    const video = videoRef.current;
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    onRecognizeFrame(canvas.toDataURL("image/jpeg", 0.82));
  };

  return (
    <Grid container spacing={3}>
      <Grid item lg={8} xs={12}>
        <Card>
          <CardContent>
            <Stack spacing={2.5}>
              <Stack
                direction={{ sm: "row", xs: "column" }}
                justifyContent="space-between"
                spacing={1.5}
              >
                <Box>
                  <Typography variant="h5">Live Camera Attendance</Typography>
                  <Typography color="text.secondary" sx={{ mt: 0.75 }} variant="body2">
                    The recognition engine is configured to process up to five faces in one frame,
                    compare them against the enrolled roster, and block duplicates automatically.
                  </Typography>
                </Box>

                <Stack direction="row" flexWrap="wrap" gap={1}>
                  <Chip
                    color={cameraStatus?.online ? "success" : "default"}
                    label={cameraStatus?.online ? "Camera Online" : "Camera Offline"}
                  />
                  <Chip
                    color={cameraStatus?.aiHealth?.connected ? "success" : "warning"}
                    label={
                      cameraStatus?.aiHealth?.connected
                        ? "Python AI Connected"
                        : "Recognition Offline"
                    }
                  />
                  <Chip
                    icon={<PeopleAltRoundedIcon />}
                    label={`Frame capacity: ${cameraStatus?.maxFacesPerFrame || 5}`}
                  />
                </Stack>
              </Stack>

              <Alert severity="info" sx={{ py: 0.5 }}>
                Best results come from clear frontal-to-slight-angle faces with even lighting.
                Recognition will still show unknown, low-quality, or rejected detections instead of
                silently skipping them.
              </Alert>

              <Box
                sx={{
                  position: "relative",
                  overflow: "hidden",
                  borderRadius: 2,
                  border: "1px solid",
                  borderColor: "divider",
                  bgcolor: "#020617"
                }}
              >
                <video autoPlay className="camera-video" muted playsInline ref={videoRef} />
              </Box>
              <canvas hidden ref={canvasRef} />

              <Stack direction={{ sm: "row", xs: "column" }} spacing={1.5}>
                <PrimaryButton
                  disabled={!activeSession || loading}
                  startIcon={<RadarRoundedIcon />}
                  onClick={scanFrame}
                >
                  Scan Current Frame
                </PrimaryButton>
                <SecondaryButton
                  disabled={!activeSession || loading}
                  startIcon={<StopCircleRoundedIcon />}
                  onClick={() => onStopSession(activeSession?.id)}
                >
                  Stop Session
                </SecondaryButton>
                <Stack direction="row" alignItems="center" spacing={1}>
                  <Switch
                    checked={autoScan}
                    disabled={!activeSession}
                    onChange={(event) => setAutoScan(event.target.checked)}
                  />
                  <Typography variant="body2">Auto Scan</Typography>
                </Stack>
              </Stack>

              {activeSession ? (
                <Stack direction={{ md: "row", xs: "column" }} spacing={1.25}>
                  <Chip
                    icon={<BoltRoundedIcon />}
                    label={`Engine: ${getEngineLabel(activeSession.engine)}`}
                  />
                  <Chip label={`Recognized: ${activeSession.recognizedCount || 0}`} />
                  <Chip label={`Duplicates: ${activeSession.duplicateCount || 0}`} />
                  <Chip label={`Unknown: ${activeSession.unknownCount || 0}`} />
                  <Chip label={`Roster: ${rosterCount}`} />
                </Stack>
              ) : null}
            </Stack>
          </CardContent>
        </Card>
      </Grid>

      <Grid item lg={4} xs={12}>
        <Stack spacing={3}>
          <Card>
            <CardContent>
              <Stack spacing={2}>
                <Typography variant="h6">Session Control</Typography>
                <TextField
                  fullWidth
                  label="Session Title"
                  value={sessionForm.title}
                  onChange={(event) => onSessionFormChange("title", event.target.value)}
                />
                <TextField
                  fullWidth
                  label="Subject"
                  value={sessionForm.subject}
                  onChange={(event) => onSessionFormChange("subject", event.target.value)}
                />
                <TextField
                  fullWidth
                  label="Department"
                  select
                  value={sessionForm.departmentId}
                  onChange={(event) => onSessionFormChange("departmentId", event.target.value)}
                >
                  {departments.map((department) => (
                    <MenuItem key={department.id} value={department.id}>
                      {department.name}
                    </MenuItem>
                  ))}
                </TextField>

                <Grid container spacing={2}>
                  <Grid item sm={4} xs={12}>
                    <TextField
                      fullWidth
                      label="Semester"
                      select
                      value={sessionForm.semester}
                      onChange={(event) => onSessionFormChange("semester", event.target.value)}
                    >
                      {SEMESTER_OPTIONS.map((semester) => (
                        <MenuItem key={semester} value={semester}>
                          {semester}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                  <Grid item sm={4} xs={12}>
                    <TextField
                      fullWidth
                      label="Section"
                      select
                      value={sessionForm.section}
                      onChange={(event) => onSessionFormChange("section", event.target.value)}
                    >
                      {SECTION_OPTIONS.map((section) => (
                        <MenuItem key={section} value={section}>
                          {section}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                  <Grid item sm={4} xs={12}>
                    <TextField
                      fullWidth
                      label="Engine"
                      select
                      value={sessionForm.engine}
                      onChange={(event) => onSessionFormChange("engine", event.target.value)}
                    >
                      {ENGINE_OPTIONS.map((engine) => (
                        <MenuItem key={engine.value} value={engine.value}>
                          {engine.label}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                </Grid>

                <PrimaryButton
                  disabled={loading || Boolean(activeSession)}
                  startIcon={<PlayCircleRoundedIcon />}
                  onClick={onStartSession}
                >
                  Start Session
                </PrimaryButton>

                <Divider />

                <Typography color="text.secondary" variant="body2">
                  {rosterCount} students in the selected academic roster.
                </Typography>
                <Typography color="text.secondary" variant="body2">
                  Last heartbeat: {formatDateTime(cameraStatus?.lastHeartbeat)}
                </Typography>
              </Stack>
            </CardContent>
          </Card>

          <Card>
            <CardContent>
              <Stack spacing={2}>
                <Typography variant="h6">Manual Backup</Typography>
                <Typography color="text.secondary" variant="body2">
                  If a student is missed by the detector, mark them from the current roster without
                  leaving the live session.
                </Typography>
                <TextField
                  fullWidth
                  disabled={!activeSession}
                  label="Student"
                  select
                  value={manualState.studentId}
                  onChange={(event) => onManualStateChange(event.target.value)}
                >
                  {eligibleStudents.map((student) => (
                    <MenuItem key={student.id} value={student.id}>
                      {student.fullName} ({student.rollNumber})
                    </MenuItem>
                  ))}
                </TextField>
                <PrimaryButton
                  disabled={!activeSession || !manualState.studentId || loading}
                  onClick={onManualMark}
                >
                  Mark Selected Student
                </PrimaryButton>
              </Stack>
            </CardContent>
          </Card>

          <Card>
            <CardContent>
              <Typography variant="h6">Recognition Feed</Typography>
              <List dense sx={{ mt: 1 }}>
                {recognitions.length ? (
                  recognitions.map((result, index) => (
                    <ListItem
                      divider
                      key={`${result.studentId || "unknown"}-${result.label}-${index}`}
                      sx={{ alignItems: "flex-start", gap: 1.25 }}
                    >
                      <Chip color={getRecognitionTone(result)} label={result.attendanceStatus} size="small" />
                      <ListItemText
                        primary={result.label || result.attendance?.studentName || "Unknown"}
                        secondary={`${result.message} (${Math.round(
                          (result.confidence || 0) * 100
                        )}% confidence, quality ${Math.round((result.quality || 0) * 100)}%)`}
                      />
                    </ListItem>
                  ))
                ) : (
                  <ListItem>
                    <ListItemText
                      primary="Waiting for detections"
                      secondary="Recognition results for every face in the current frame will appear here."
                    />
                  </ListItem>
                )}
              </List>
            </CardContent>
          </Card>
        </Stack>
      </Grid>
    </Grid>
  );
}
