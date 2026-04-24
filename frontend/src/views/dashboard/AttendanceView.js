import { useCallback, useEffect, useMemo, useState } from "react";
import dayjs from "dayjs";
import {
  Alert,
  Card,
  CardContent,
  Chip,
  Grid,
  MenuItem,
  Stack,
  TextField,
  Typography
} from "@mui/material";
import AttendanceStats from "@/components/AttendanceStats";
import CustomDataGrid from "@/components/CustomDataGrid";
import SearchFilters from "@/components/SearchFilters";
import { PrimaryButton } from "@/components/CustomButtons";
import attendanceService from "@/services/attendanceService";
import studentService from "@/services/studentService";
import useAuth from "@/hooks/useAuth";
import { useAppSnackbar } from "@/context/SnackbarContext";
import { SECTION_OPTIONS, SEMESTER_OPTIONS } from "@/utils/constants";
import { formatDateTime } from "@/utils/formatters";

const initialFilters = {
  date: dayjs().format("YYYY-MM-DD"),
  departmentId: "",
  semester: "",
  section: ""
};

const initialManualState = {
  sessionId: "",
  studentId: ""
};

export default function AttendanceView() {
  const { user } = useAuth();
  const { showSnackbar } = useAppSnackbar();
  const [filters, setFilters] = useState(initialFilters);
  const [attendancePayload, setAttendancePayload] = useState({
    rows: [],
    summary: {}
  });
  const [sessions, setSessions] = useState([]);
  const [departments, setDepartments] = useState([]);
  const [students, setStudents] = useState([]);
  const [loading, setLoading] = useState(true);
  const [manualState, setManualState] = useState(initialManualState);
  const [manualBusy, setManualBusy] = useState(false);

  const loadAttendance = useCallback(async () => {
    try {
      setLoading(true);
      const [attendanceData, sessionsData, studentData] = await Promise.all([
        attendanceService.getRecords(filters),
        attendanceService.getSessions(),
        studentService.getStudents()
      ]);

      setAttendancePayload(attendanceData);
      setSessions(sessionsData.rows);
      setDepartments(studentData.departments);
      setStudents(studentData.rows);
      setManualState((currentValue) => ({
        sessionId:
          currentValue.sessionId ||
          sessionsData.activeSession?.id ||
          sessionsData.rows.find((session) => session.status === "active")?.id ||
          "",
        studentId: currentValue.studentId
      }));
    } catch (error) {
      showSnackbar(error.message || "Unable to load attendance data.", "error");
    } finally {
      setLoading(false);
    }
  }, [filters, showSnackbar]);

  useEffect(() => {
    loadAttendance();
  }, [loadAttendance]);

  const selectedSession = useMemo(
    () => sessions.find((session) => session.id === manualState.sessionId) || null,
    [manualState.sessionId, sessions]
  );

  const eligibleStudents = useMemo(() => {
    if (!selectedSession) {
      return [];
    }

    return students.filter(
      (student) =>
        student.departmentId === selectedSession.departmentId &&
        student.semester === selectedSession.semester &&
        student.section === selectedSession.section
    );
  }, [selectedSession, students]);

  const handleManualAttendance = async () => {
    try {
      setManualBusy(true);
      const result = await attendanceService.markManual({
        ...manualState,
        actor: user
      });

      await loadAttendance();
      showSnackbar(result.message || "Attendance updated successfully.");
    } catch (error) {
      showSnackbar(error.message || "Unable to update attendance manually.", "error");
    } finally {
      setManualBusy(false);
    }
  };

  return (
    <Stack spacing={3}>
      <SearchFilters
        fields={[
          {
            key: "date",
            label: "Attendance Date",
            type: "date",
            lg: 3
          },
          {
            key: "departmentId",
            label: "Department",
            type: "select",
            options: departments.map((department) => ({
              label: department.name,
              value: department.id
            }))
          },
          {
            key: "semester",
            label: "Semester",
            type: "select",
            options: SEMESTER_OPTIONS.map((semester) => ({
              label: `Semester ${semester}`,
              value: semester
            }))
          },
          {
            key: "section",
            label: "Section",
            type: "select",
            options: SECTION_OPTIONS.map((section) => ({
              label: `Section ${section}`,
              value: section
            }))
          }
        ]}
        filters={filters}
        onChange={(key, value) =>
          setFilters((currentValue) => ({
            ...currentValue,
            [key]: value
          }))
        }
        onReset={() => setFilters(initialFilters)}
      />

      <Grid container spacing={3}>
        <Grid item lg={8} xs={12}>
          <AttendanceStats summary={attendancePayload.summary} />
        </Grid>
        <Grid item lg={4} xs={12}>
          <Card sx={{ height: "100%" }}>
            <CardContent>
              <Stack spacing={2}>
                <Typography variant="h6">Manual Attendance Backup</Typography>
                <Typography color="text.secondary" variant="body2">
                  Use this when face detection misses a student or the camera feed is not reliable.
                </Typography>

                <Alert severity="warning" sx={{ py: 0.5 }}>
                  Manual marking uses the same active session rules, so only students from the
                  session roster can be added.
                </Alert>

                <TextField
                  fullWidth
                  label="Session"
                  select
                  value={manualState.sessionId}
                  onChange={(event) =>
                    setManualState({
                      sessionId: event.target.value,
                      studentId: ""
                    })
                  }
                >
                  {sessions.map((session) => (
                    <MenuItem key={session.id} value={session.id}>
                      {session.title} ({session.status})
                    </MenuItem>
                  ))}
                </TextField>

                <TextField
                  fullWidth
                  disabled={!selectedSession}
                  label="Student"
                  select
                  value={manualState.studentId}
                  onChange={(event) =>
                    setManualState((currentValue) => ({
                      ...currentValue,
                      studentId: event.target.value
                    }))
                  }
                >
                  {eligibleStudents.map((student) => (
                    <MenuItem key={student.id} value={student.id}>
                      {student.fullName} ({student.rollNumber})
                    </MenuItem>
                  ))}
                </TextField>

                {selectedSession ? (
                  <Stack direction="row" flexWrap="wrap" gap={1}>
                    <Chip label={selectedSession.departmentCode} size="small" />
                    <Chip label={`Semester ${selectedSession.semester}`} size="small" />
                    <Chip label={`Section ${selectedSession.section}`} size="small" />
                  </Stack>
                ) : null}

                <PrimaryButton
                  disabled={!manualState.sessionId || !manualState.studentId || manualBusy}
                  onClick={handleManualAttendance}
                >
                  {manualBusy ? "Updating..." : "Mark Attendance Manually"}
                </PrimaryButton>
              </Stack>
            </CardContent>
          </Card>
        </Grid>
      </Grid>

      <CustomDataGrid
        columns={[
          { field: "title", headerName: "Session", minWidth: 220, flex: 1.15 },
          { field: "departmentName", headerName: "Department", minWidth: 160, flex: 0.85 },
          { field: "subject", headerName: "Subject", minWidth: 160, flex: 0.95 },
          {
            field: "status",
            headerName: "Status",
            minWidth: 140,
            flex: 0.7,
            render: (row) => (
              <Chip
                color={row.status === "active" ? "success" : "default"}
                label={row.status}
                size="small"
              />
            )
          },
          { field: "recognizedCount", headerName: "Recognized", minWidth: 130, flex: 0.6 },
          { field: "duplicateCount", headerName: "Duplicates", minWidth: 130, flex: 0.6 },
          { field: "unknownCount", headerName: "Unknown", minWidth: 120, flex: 0.6 },
          {
            field: "startedAt",
            headerName: "Started At",
            minWidth: 190,
            flex: 0.9,
            render: (row) => formatDateTime(row.startedAt)
          }
        ]}
        loading={loading}
        rows={sessions}
        title="Attendance Sessions"
        subtitle="Track live and completed sessions, recognition volume, and manual fallback coverage."
      />

      <CustomDataGrid
        columns={[
          { field: "studentName", headerName: "Student", minWidth: 220, flex: 1.15 },
          { field: "rollNumber", headerName: "Roll Number", minWidth: 160, flex: 0.8 },
          { field: "departmentName", headerName: "Department", minWidth: 160, flex: 0.85 },
          { field: "subject", headerName: "Subject", minWidth: 160, flex: 0.9 },
          {
            field: "source",
            headerName: "Source",
            minWidth: 150,
            flex: 0.75,
            render: (row) => (
              <Chip
                color={String(row.source || "").startsWith("manual") ? "warning" : "info"}
                label={row.source}
                size="small"
              />
            )
          },
          {
            field: "confidence",
            headerName: "Confidence",
            minWidth: 140,
            flex: 0.7,
            render: (row) => `${Math.round((row.confidence || 0) * 100)}%`
          },
          {
            field: "recognizedAt",
            headerName: "Recognized At",
            minWidth: 190,
            flex: 0.9,
            render: (row) => formatDateTime(row.recognizedAt)
          }
        ]}
        loading={loading}
        rows={attendancePayload.rows}
        title="Attendance Records"
        subtitle="Review AI and manual attendance evidence for the selected day."
      />
    </Stack>
  );
}
