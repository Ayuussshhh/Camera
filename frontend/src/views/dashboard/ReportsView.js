import { useEffect, useState } from "react";
import {
  Chip,
  Stack
} from "@mui/material";
import dayjs from "dayjs";
import ReportCharts from "@/components/ReportCharts";
import SearchFilters from "@/components/SearchFilters";
import CustomDataGrid from "@/components/CustomDataGrid";
import attendanceService from "@/services/attendanceService";
import reportService from "@/services/reportService";
import studentService from "@/services/studentService";
import { PrimaryButton, SecondaryButton } from "@/components/CustomButtons";
import { useAppSnackbar } from "@/context/SnackbarContext";
import { exportRowsToCsv, exportRowsToExcel, exportRowsToPdf } from "@/utils/exportHelpers";
import {
  SECTION_OPTIONS,
  SEMESTER_OPTIONS
} from "@/utils/constants";
import { formatDateTime } from "@/utils/formatters";

const initialFilters = {
  date: dayjs().format("YYYY-MM-DD"),
  departmentId: "",
  semester: "",
  section: ""
};

export default function ReportsView() {
  const { showSnackbar } = useAppSnackbar();
  const [filters, setFilters] = useState(initialFilters);
  const [analytics, setAnalytics] = useState(null);
  const [records, setRecords] = useState([]);
  const [departments, setDepartments] = useState([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadReports() {
      try {
        setLoading(true);
        const [analyticsPayload, recordsPayload, studentPayload] = await Promise.all([
          reportService.getAnalytics(filters),
          attendanceService.getRecords(filters),
          studentService.getStudents()
        ]);

        setAnalytics(analyticsPayload);
        setRecords(recordsPayload.rows);
        setDepartments(studentPayload.departments);
      } catch (error) {
        showSnackbar(error.message || "Unable to load reports.", "error");
      } finally {
        setLoading(false);
      }
    }

    loadReports();
  }, [filters, showSnackbar]);

  const exportRows = records.map((record) => ({
    Student: record.studentName,
    RollNumber: record.rollNumber,
    Department: record.departmentName,
    Subject: record.subject,
    Confidence: `${Math.round((record.confidence || 0) * 100)}%`,
    RecognizedAt: formatDateTime(record.recognizedAt)
  }));

  return (
    <Stack spacing={3}>
      <SearchFilters
        actions={
          <Stack direction={{ sm: "row", xs: "column" }} spacing={1.5}>
            <SecondaryButton onClick={() => exportRowsToCsv("attendance-report", exportRows)}>
              Export CSV
            </SecondaryButton>
            <SecondaryButton
              onClick={() =>
                exportRowsToExcel("attendance-report", exportRows, "Attendance Report")
              }
            >
              Export Excel
            </SecondaryButton>
            <PrimaryButton
              onClick={() =>
                exportRowsToPdf(
                  "attendance-report",
                  "Attendance Report",
                  Object.keys(exportRows[0] || {}),
                  exportRows
                )
              }
            >
              Export PDF
            </PrimaryButton>
          </Stack>
        }
        fields={[
          {
            key: "date",
            label: "Report Date",
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

      <ReportCharts analytics={analytics} />

      <CustomDataGrid
        columns={[
          { field: "studentName", headerName: "Student", minWidth: 220, flex: 1.1 },
          { field: "departmentName", headerName: "Department", minWidth: 160, flex: 0.85 },
          { field: "subject", headerName: "Subject", minWidth: 160, flex: 0.9 },
          {
            field: "confidence",
            headerName: "Confidence",
            minWidth: 150,
            flex: 0.7,
            render: (row) => (
              <Chip
                color={row.confidence >= 0.9 ? "success" : "warning"}
                label={`${Math.round((row.confidence || 0) * 100)}%`}
                size="small"
              />
            )
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
        rows={records}
        title="Attendance Export Preview"
        subtitle="Preview the exact records that will be exported for institutional reporting."
      />

      <CustomDataGrid
        columns={[
          { field: "actorName", headerName: "Actor", minWidth: 160, flex: 0.8 },
          { field: "action", headerName: "Action", minWidth: 150, flex: 0.75 },
          { field: "entityType", headerName: "Entity", minWidth: 140, flex: 0.7 },
          { field: "message", headerName: "Message", minWidth: 260, flex: 1.4 },
          {
            field: "createdAt",
            headerName: "Timestamp",
            minWidth: 190,
            flex: 0.9,
            render: (row) => formatDateTime(row.createdAt)
          }
        ]}
        loading={loading}
        rows={analytics?.auditLogs || []}
        title="Audit Trail"
        subtitle="Trace report access, attendance actions, and system-level changes."
      />
    </Stack>
  );
}
