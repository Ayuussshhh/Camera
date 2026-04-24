import { useCallback, useEffect, useState } from "react";
import {
  Avatar,
  Chip,
  IconButton,
  Stack,
  Tooltip,
  Typography
} from "@mui/material";
import EditRoundedIcon from "@mui/icons-material/EditRounded";
import SearchFilters from "@/components/SearchFilters";
import StudentForm from "@/components/StudentForm";
import CustomDataGrid from "@/components/CustomDataGrid";
import { PrimaryButton } from "@/components/CustomButtons";
import studentService from "@/services/studentService";
import useAuth from "@/hooks/useAuth";
import { useAppSnackbar } from "@/context/SnackbarContext";
import { SECTION_OPTIONS, SEMESTER_OPTIONS } from "@/utils/constants";
import { formatDateTime } from "@/utils/formatters";

const initialFilters = {
  search: "",
  departmentId: "",
  semester: "",
  section: ""
};

export default function StudentsView() {
  const { user } = useAuth();
  const { showSnackbar } = useAppSnackbar();
  const [filters, setFilters] = useState(initialFilters);
  const [students, setStudents] = useState([]);
  const [departments, setDepartments] = useState([]);
  const [loading, setLoading] = useState(true);
  const [submitting, setSubmitting] = useState(false);
  const [formOpen, setFormOpen] = useState(false);
  const [editingStudent, setEditingStudent] = useState(null);

  const loadStudents = useCallback(async () => {
    setLoading(true);

    try {
      const payload = await studentService.getStudents(filters);
      setStudents(payload.rows);
      setDepartments(payload.departments);
    } catch (error) {
      showSnackbar(error.message || "Unable to load students.", "error");
    } finally {
      setLoading(false);
    }
  }, [filters, showSnackbar]);

  useEffect(() => {
    loadStudents();
  }, [loadStudents]);

  const closeForm = () => {
    setFormOpen(false);
    setEditingStudent(null);
  };

  const handleCreateStudent = async (payload) => {
    try {
      setSubmitting(true);
      const student = await studentService.createStudent({
        ...payload,
        actor: user
      });

      if (payload.capturedImages?.length) {
        await studentService.enrollFace({
          studentId: student.id,
          images: payload.capturedImages,
          engine: payload.engine,
          actor: user
        });
      }

      await loadStudents();
      closeForm();
      showSnackbar("Student registered successfully.");
    } catch (error) {
      showSnackbar(error.message || "Student registration failed.", "error");
    } finally {
      setSubmitting(false);
    }
  };

  const handleUpdateStudent = async (payload) => {
    try {
      setSubmitting(true);
      await studentService.updateStudent(editingStudent.id, {
        ...payload,
        actor: user
      });

      if (payload.capturedImages?.length) {
        await studentService.enrollFace({
          studentId: editingStudent.id,
          images: payload.capturedImages,
          engine: payload.engine,
          actor: user
        });
      }

      await loadStudents();
      closeForm();
      showSnackbar("Student updated successfully.");
    } catch (error) {
      showSnackbar(error.message || "Student update failed.", "error");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Stack spacing={3}>
      <SearchFilters
        actions={<PrimaryButton onClick={() => setFormOpen(true)}>Register Student</PrimaryButton>}
        fields={[
          {
            key: "search",
            label: "Search",
            placeholder: "Name, roll number, email",
            type: "text",
            lg: 4
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

      <CustomDataGrid
        columns={[
          {
            field: "fullName",
            headerName: "Student",
            minWidth: 250,
            flex: 1.2,
            render: (row) => (
              <Stack direction="row" spacing={1.5} sx={{ alignItems: "center" }}>
                <Avatar sx={{ bgcolor: "primary.main" }}>{row.fullName?.[0] || "S"}</Avatar>
                <Stack spacing={0.2}>
                  <Typography sx={{ fontWeight: 600 }} variant="body2">
                    {row.fullName}
                  </Typography>
                  <Typography color="text.secondary" variant="caption">
                    {row.email}
                  </Typography>
                </Stack>
              </Stack>
            )
          },
          { field: "rollNumber", headerName: "Roll Number", minWidth: 150, flex: 0.75 },
          { field: "departmentCode", headerName: "Department", minWidth: 130, flex: 0.65 },
          { field: "semester", headerName: "Semester", minWidth: 120, flex: 0.55 },
          { field: "section", headerName: "Section", minWidth: 100, flex: 0.45 },
          {
            field: "faceEnrollmentStatus",
            headerName: "Face Profile",
            minWidth: 210,
            flex: 0.95,
            render: (row) => (
              <Stack direction="row" spacing={1} sx={{ alignItems: "center" }}>
                <Chip
                  color={row.faceEnrollmentStatus === "completed" ? "success" : "warning"}
                  label={row.faceEnrollmentStatus}
                  size="small"
                />
                <Typography color="text.secondary" variant="caption">
                  {row.faceImageCount || 0} images
                </Typography>
              </Stack>
            )
          },
          {
            field: "lastTrainedAt",
            headerName: "Last Trained",
            minWidth: 180,
            flex: 0.85,
            render: (row) => formatDateTime(row.lastTrainedAt)
          },
          {
            field: "actions",
            headerName: "Actions",
            minWidth: 110,
            flex: 0.45,
            sortable: false,
            render: (row) => (
              <Tooltip title="Edit student">
                <IconButton
                  color="primary"
                  onClick={() => {
                    setEditingStudent({
                      ...row,
                      engine: "mediapipe"
                    });
                    setFormOpen(true);
                  }}
                >
                  <EditRoundedIcon fontSize="small" />
                </IconButton>
              </Tooltip>
            )
          }
        ]}
        loading={loading}
        rows={students}
        title="Student Directory"
        subtitle="Create, edit, and re-enroll student profiles with clearer face-training visibility."
      />

      <StudentForm
        departments={departments}
        initialValues={editingStudent}
        loading={submitting}
        mode={editingStudent ? "edit" : "create"}
        open={formOpen}
        onClose={closeForm}
        onSubmit={editingStudent ? handleUpdateStudent : handleCreateStudent}
      />
    </Stack>
  );
}
