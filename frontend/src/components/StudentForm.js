import { useEffect, useMemo, useState } from "react";
import {
  Alert,
  Box,
  Dialog,
  DialogActions,
  DialogContent,
  DialogTitle,
  Divider,
  Grid,
  MenuItem,
  Stack,
  TextField,
  Typography
} from "@mui/material";
import FaceCaptureBox from "@/components/FaceCaptureBox";
import { PrimaryButton, SecondaryButton } from "@/components/CustomButtons";
import {
  ENGINE_OPTIONS,
  SECTION_OPTIONS,
  SEMESTER_OPTIONS
} from "@/utils/constants";

const EMPTY_FORM = {
  fullName: "",
  email: "",
  rollNumber: "",
  guardianPhone: "",
  departmentId: "",
  semester: "",
  section: "",
  engine: "mediapipe"
};

export default function StudentForm({
  open,
  loading,
  departments = [],
  mode = "create",
  initialValues = null,
  onClose,
  onSubmit
}) {
  const [formState, setFormState] = useState(EMPTY_FORM);
  const [capturedImages, setCapturedImages] = useState([]);

  const isEditMode = mode === "edit";
  const title = isEditMode ? "Edit Student" : "Register Student";
  const actionLabel = isEditMode ? "Save Changes" : "Register & Train";

  useEffect(() => {
    if (!open) {
      return;
    }

    setFormState({
      ...EMPTY_FORM,
      ...(initialValues || {}),
      departmentId:
        initialValues?.departmentId || departments[0]?.id || EMPTY_FORM.departmentId
    });
    setCapturedImages([]);
  }, [departments, initialValues, open]);

  const canSubmit = useMemo(
    () =>
      Boolean(
        formState.fullName &&
          formState.email &&
          formState.rollNumber &&
          formState.departmentId &&
          formState.semester &&
          formState.section
      ),
    [formState]
  );

  const handleChange = (key, value) => {
    setFormState((currentValue) => ({
      ...currentValue,
      [key]: value
    }));
  };

  const handleSubmit = () => {
    onSubmit({
      ...formState,
      capturedImages
    });
  };

  return (
    <Dialog fullWidth maxWidth="lg" open={open} onClose={onClose}>
      <DialogTitle sx={{ pb: 1.5 }}>
        <Stack spacing={0.75}>
          <Typography variant="h5">{title}</Typography>
          <Typography color="text.secondary" variant="body2">
            Manage academic details, enrollment readiness, and optional face profile refresh in one
            place.
          </Typography>
        </Stack>
      </DialogTitle>
      <DialogContent dividers sx={{ p: 0 }}>
        <Grid container>
          <Grid item md={7} xs={12}>
            <Box sx={{ p: 3 }}>
              <Stack spacing={3}>
                <Alert severity="info" sx={{ alignItems: "center" }}>
                  Upload 3 to 5 clear images. Five images with slight head-angle changes give the
                  strongest face embedding and more stable attendance marking.
                </Alert>

                <Grid container spacing={2}>
                  <Grid item md={6} xs={12}>
                    <TextField
                      fullWidth
                      label="Full Name"
                      required
                      value={formState.fullName}
                      onChange={(event) => handleChange("fullName", event.target.value)}
                    />
                  </Grid>
                  <Grid item md={6} xs={12}>
                    <TextField
                      fullWidth
                      label="Email"
                      required
                      value={formState.email}
                      onChange={(event) => handleChange("email", event.target.value)}
                    />
                  </Grid>
                  <Grid item md={4} xs={12}>
                    <TextField
                      fullWidth
                      label="Roll Number"
                      required
                      value={formState.rollNumber}
                      onChange={(event) => handleChange("rollNumber", event.target.value)}
                    />
                  </Grid>
                  <Grid item md={4} xs={12}>
                    <TextField
                      fullWidth
                      label="Guardian Phone"
                      value={formState.guardianPhone}
                      onChange={(event) => handleChange("guardianPhone", event.target.value)}
                    />
                  </Grid>
                  <Grid item md={4} xs={12}>
                    <TextField
                      fullWidth
                      label="Recognition Engine"
                      select
                      value={formState.engine}
                      onChange={(event) => handleChange("engine", event.target.value)}
                    >
                      {ENGINE_OPTIONS.map((engine) => (
                        <MenuItem key={engine.value} value={engine.value}>
                          {engine.label}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                  <Grid item md={4} xs={12}>
                    <TextField
                      fullWidth
                      label="Department"
                      required
                      select
                      value={formState.departmentId}
                      onChange={(event) => handleChange("departmentId", event.target.value)}
                    >
                      {departments.map((department) => (
                        <MenuItem key={department.id} value={department.id}>
                          {department.name}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                  <Grid item md={4} xs={12}>
                    <TextField
                      fullWidth
                      label="Semester"
                      required
                      select
                      value={formState.semester}
                      onChange={(event) => handleChange("semester", event.target.value)}
                    >
                      {SEMESTER_OPTIONS.map((semester) => (
                        <MenuItem key={semester} value={semester}>
                          Semester {semester}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                  <Grid item md={4} xs={12}>
                    <TextField
                      fullWidth
                      label="Section"
                      required
                      select
                      value={formState.section}
                      onChange={(event) => handleChange("section", event.target.value)}
                    >
                      {SECTION_OPTIONS.map((section) => (
                        <MenuItem key={section} value={section}>
                          Section {section}
                        </MenuItem>
                      ))}
                    </TextField>
                  </Grid>
                </Grid>
              </Stack>
            </Box>
          </Grid>

          <Grid item md={5} xs={12}>
            <Box sx={{ p: 3, height: "100%", bgcolor: "background.default" }}>
              <Stack spacing={2.5}>
                <Box>
                  <Typography variant="h6">
                    {isEditMode ? "Refresh Face Profile" : "Face Enrollment"}
                  </Typography>
                  <Typography color="text.secondary" sx={{ mt: 0.75 }} variant="body2">
                    {isEditMode
                      ? "Leave this empty if you only want to update student details. Add new samples to retrain the profile."
                      : "Capture clean samples from the camera or upload them directly to build the student embedding."}
                  </Typography>
                </Box>

                <Divider />

                <FaceCaptureBox maxImages={5} onImagesChange={setCapturedImages} />
              </Stack>
            </Box>
          </Grid>
        </Grid>
      </DialogContent>
      <DialogActions sx={{ p: 2.5 }}>
        <SecondaryButton onClick={onClose}>Cancel</SecondaryButton>
        <PrimaryButton disabled={loading || !canSubmit} onClick={handleSubmit}>
          {loading ? "Saving..." : actionLabel}
        </PrimaryButton>
      </DialogActions>
    </Dialog>
  );
}
