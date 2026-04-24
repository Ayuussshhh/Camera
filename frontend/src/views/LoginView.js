import { useEffect, useState } from "react";
import { useRouter } from "next/router";
import {
  Box,
  Card,
  CardContent,
  Chip,
  Grid,
  MenuItem,
  Stack,
  TextField,
  Typography
} from "@mui/material";
import SecurityRoundedIcon from "@mui/icons-material/SecurityRounded";
import HistoryEduRoundedIcon from "@mui/icons-material/HistoryEduRounded";
import RadarRoundedIcon from "@mui/icons-material/RadarRounded";
import VerifiedUserRoundedIcon from "@mui/icons-material/VerifiedUserRounded";
import useAuth from "@/hooks/useAuth";
import { useAppSnackbar } from "@/context/SnackbarContext";
import { PrimaryButton } from "@/components/CustomButtons";
import { DEFAULT_LOGIN_HINTS, USER_ROLES } from "@/utils/constants";

const capabilityItems = [
  {
    title: "Live recognition operations",
    description: "Start sessions, scan frames, and review multi-face detections in one workflow.",
    icon: <RadarRoundedIcon fontSize="small" />
  },
  {
    title: "Verified attendance trail",
    description: "Every recognition and manual correction stays visible for review and reporting.",
    icon: <HistoryEduRoundedIcon fontSize="small" />
  },
  {
    title: "Controlled access",
    description: "Role-based sign-in keeps admin and teacher workflows clean and accountable.",
    icon: <VerifiedUserRoundedIcon fontSize="small" />
  }
];

export default function LoginView() {
  const router = useRouter();
  const { login, user, loading } = useAuth();
  const { showSnackbar } = useAppSnackbar();
  const [submitting, setSubmitting] = useState(false);
  const [formState, setFormState] = useState({
    role: "admin",
    email: "admin@college.edu",
    password: "admin123"
  });

  useEffect(() => {
    if (!loading && user) {
      router.replace("/dashboard");
    }
  }, [loading, router, user]);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setSubmitting(true);

    try {
      await login(formState);
      showSnackbar("Login successful.");
      router.replace("/dashboard");
    } catch (error) {
      showSnackbar(error.message || "Unable to authenticate user.", "error");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <Grid container sx={{ minHeight: "100vh", bgcolor: "#F3F6FB" }}>
      <Grid
        item
        lg={7}
        xs={12}
        sx={{
          px: { lg: 8, md: 5, xs: 3 },
          py: { lg: 6, md: 5, xs: 4 },
          display: "flex",
          alignItems: "center"
        }}
      >
        <Stack spacing={4} sx={{ width: "100%" }}>
          <Stack spacing={2.5}>
            <Chip
              icon={<SecurityRoundedIcon />}
              label="FaceTrace Attendance Console"
              sx={{
                alignSelf: "flex-start",
                bgcolor: "rgba(18, 52, 86, 0.08)",
                color: "primary.dark",
                fontWeight: 700
              }}
            />
            <Box>
              <Typography
                sx={{ color: "#0F172A", fontSize: { md: "3.15rem", xs: "2.3rem" }, fontWeight: 700 }}
                variant="h1"
              >
                Attendance operations for real classrooms.
              </Typography>
              <Typography color="text.secondary" sx={{ maxWidth: 680, mt: 2 }} variant="h6">
                Run student enrollment, live face-based attendance, manual corrections, and audit
                review from one professional control surface.
              </Typography>
            </Box>
          </Stack>

          <Grid container spacing={2}>
            {capabilityItems.map((item) => (
              <Grid item md={4} sm={6} xs={12} key={item.title}>
                <Card
                  sx={{
                    height: "100%",
                    borderColor: "rgba(15, 23, 42, 0.08)",
                    boxShadow: "0 18px 40px rgba(15, 23, 42, 0.06)"
                  }}
                >
                  <CardContent sx={{ p: 3 }}>
                    <Stack spacing={1.5}>
                      <Box
                        sx={{
                          width: 42,
                          height: 42,
                          borderRadius: 2,
                          display: "grid",
                          placeItems: "center",
                          bgcolor: "rgba(23, 105, 170, 0.1)",
                          color: "primary.main"
                        }}
                      >
                        {item.icon}
                      </Box>
                      <Typography sx={{ color: "#0F172A", fontWeight: 700 }} variant="subtitle1">
                        {item.title}
                      </Typography>
                      <Typography color="text.secondary" variant="body2">
                        {item.description}
                      </Typography>
                    </Stack>
                  </CardContent>
                </Card>
              </Grid>
            ))}
          </Grid>
        </Stack>
      </Grid>

      <Grid
        item
        lg={5}
        xs={12}
        sx={{
          background:
            "linear-gradient(180deg, rgba(12,25,43,0.98) 0%, rgba(18,52,86,0.96) 48%, rgba(31,91,145,0.95) 100%)",
          color: "#E2E8F0",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          px: 3,
          py: { md: 5, xs: 4 }
        }}
      >
        <Card
          sx={{
            maxWidth: 460,
            width: "100%",
            bgcolor: "rgba(255,255,255,0.96)",
            boxShadow: "0 24px 60px rgba(2, 6, 23, 0.28)"
          }}
        >
          <CardContent sx={{ p: { sm: 4.5, xs: 3 } }}>
            <Stack spacing={3}>
              <Box>
                <Typography sx={{ color: "#0F172A", fontWeight: 700 }} variant="h4">
                  Secure Login
                </Typography>
                <Typography color="text.secondary" sx={{ mt: 1 }} variant="body2">
                  Sign in as an administrator or teacher to manage attendance sessions and student
                  enrollment.
                </Typography>
              </Box>

              <Stack direction="row" flexWrap="wrap" gap={1}>
                {DEFAULT_LOGIN_HINTS.map((hint) => (
                  <Chip
                    clickable
                    key={hint.role}
                    label={`${hint.role}: ${hint.email}`}
                    onClick={() => setFormState(hint)}
                    sx={{ fontWeight: 600 }}
                  />
                ))}
              </Stack>

              <Box component="form" onSubmit={handleSubmit}>
                <Stack spacing={2}>
                  <TextField
                    fullWidth
                    label="Role"
                    select
                    value={formState.role}
                    onChange={(event) =>
                      setFormState((currentValue) => ({
                        ...currentValue,
                        role: event.target.value
                      }))
                    }
                  >
                    {USER_ROLES.map((role) => (
                      <MenuItem key={role.value} value={role.value}>
                        {role.label}
                      </MenuItem>
                    ))}
                  </TextField>
                  <TextField
                    fullWidth
                    label="Email"
                    value={formState.email}
                    onChange={(event) =>
                      setFormState((currentValue) => ({
                        ...currentValue,
                        email: event.target.value
                      }))
                    }
                  />
                  <TextField
                    fullWidth
                    label="Password"
                    type="password"
                    value={formState.password}
                    onChange={(event) =>
                      setFormState((currentValue) => ({
                        ...currentValue,
                        password: event.target.value
                      }))
                    }
                  />
                  <PrimaryButton disabled={submitting} size="large" type="submit">
                    {submitting ? "Signing In..." : "Enter Dashboard"}
                  </PrimaryButton>
                </Stack>
              </Box>
            </Stack>
          </CardContent>
        </Card>
      </Grid>
    </Grid>
  );
}
