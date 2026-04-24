import { useEffect, useMemo, useState } from "react";
import { Stack } from "@mui/material";
import CameraCard from "@/components/CameraCard";
import attendanceService from "@/services/attendanceService";
import cameraService from "@/services/cameraService";
import settingsService from "@/services/settingsService";
import studentService from "@/services/studentService";
import useAuth from "@/hooks/useAuth";
import useAttendancePolling from "@/hooks/useAttendancePolling";
import { useAppSnackbar } from "@/context/SnackbarContext";

export default function LiveCameraView() {
  const { user } = useAuth();
  const { showSnackbar } = useAppSnackbar();
  const [cameraStatus, setCameraStatus] = useState(null);
  const [departments, setDepartments] = useState([]);
  const [students, setStudents] = useState([]);
  const [recognitions, setRecognitions] = useState([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [manualStudentId, setManualStudentId] = useState("");
  const [sessionForm, setSessionForm] = useState({
    title: "Morning Attendance Session",
    subject: "Artificial Intelligence",
    departmentId: "",
    semester: "6",
    section: "A",
    engine: "mediapipe"
  });

  const loadStatus = async () => {
    const status = await cameraService.getStatus();
    setCameraStatus(status);
  };

  useAttendancePolling(
    () => {
      if (cameraStatus?.activeSession) {
        loadStatus().catch(() => undefined);
      }
    },
    Boolean(cameraStatus?.activeSession),
    5000
  );

  useEffect(() => {
    async function bootstrap() {
      try {
        setLoading(true);
        const [cameraPayload, settingsPayload, studentPayload] = await Promise.all([
          cameraService.getStatus(),
          settingsService.getSettings(),
          studentService.getStudents()
        ]);

        setCameraStatus(cameraPayload);
        setDepartments(studentPayload.departments);
        setStudents(studentPayload.rows);
        setSessionForm((currentValue) => ({
          ...currentValue,
          departmentId: currentValue.departmentId || studentPayload.departments[0]?.id || "",
          engine: settingsPayload.activeEngine || currentValue.engine
        }));
      } catch (error) {
        showSnackbar(error.message || "Unable to load live camera view.", "error");
      } finally {
        setLoading(false);
      }
    }

    bootstrap();
  }, [showSnackbar]);

  const handleStartSession = async () => {
    try {
      setBusy(true);
      await attendanceService.startSession({
        ...sessionForm,
        actor: user,
        teacherId: user.teacherId
      });
      await loadStatus();
      setManualStudentId("");
      setRecognitions([]);
      showSnackbar("Attendance session started.");
    } catch (error) {
      showSnackbar(error.message || "Unable to start session.", "error");
    } finally {
      setBusy(false);
    }
  };

  const handleStopSession = async (sessionId) => {
    try {
      setBusy(true);
      await attendanceService.stopSession(sessionId, user);
      await loadStatus();
      showSnackbar("Attendance session stopped.");
    } catch (error) {
      showSnackbar(error.message || "Unable to stop session.", "error");
    } finally {
      setBusy(false);
    }
  };

  const handleRecognizeFrame = async (frame) => {
    try {
      setBusy(true);
      const payload = await cameraService.recognizeFrame({
        frame,
        engine: cameraStatus?.activeSession?.engine || sessionForm.engine,
        actor: user
      });
      setRecognitions(payload.results || []);
      await loadStatus();
    } catch (error) {
      showSnackbar(error.message || "Frame recognition failed.", "error");
    } finally {
      setBusy(false);
    }
  };

  const activeAcademicContext = {
    departmentId: cameraStatus?.activeSession?.departmentId || sessionForm.departmentId,
    semester: cameraStatus?.activeSession?.semester || sessionForm.semester,
    section: cameraStatus?.activeSession?.section || sessionForm.section
  };

  const eligibleStudents = useMemo(
    () =>
      students.filter(
        (student) =>
          student.departmentId === activeAcademicContext.departmentId &&
          student.semester === activeAcademicContext.semester &&
          student.section === activeAcademicContext.section
      ),
    [activeAcademicContext.departmentId, activeAcademicContext.section, activeAcademicContext.semester, students]
  );

  const rosterCount = eligibleStudents.length;

  const handleManualMark = async () => {
    try {
      setBusy(true);
      const result = await attendanceService.markManual({
        sessionId: cameraStatus?.activeSession?.id,
        studentId: manualStudentId,
        actor: user
      });
      await loadStatus();
      showSnackbar(result.message || "Attendance updated manually.");
    } catch (error) {
      showSnackbar(error.message || "Manual attendance failed.", "error");
    } finally {
      setBusy(false);
    }
  };

  return (
    <Stack spacing={3}>
      <CameraCard
        cameraStatus={{
          ...cameraStatus,
          maxFacesPerFrame: 5
        }}
        departments={departments}
        eligibleStudents={eligibleStudents}
        loading={loading || busy}
        manualState={{ studentId: manualStudentId }}
        recognitions={recognitions}
        rosterCount={rosterCount}
        sessionForm={sessionForm}
        onManualMark={handleManualMark}
        onManualStateChange={setManualStudentId}
        onRecognizeFrame={handleRecognizeFrame}
        onSessionFormChange={(key, value) =>
          setSessionForm((currentValue) => ({
            ...currentValue,
            [key]: value
          }))
        }
        onStartSession={handleStartSession}
        onStopSession={handleStopSession}
      />
    </Stack>
  );
}
