import { alpha, useTheme } from "@mui/material/styles";
import {
  Box,
  Card,
  CardContent,
  Grid,
  LinearProgress,
  Stack,
  Typography
} from "@mui/material";
import HowToRegRoundedIcon from "@mui/icons-material/HowToRegRounded";
import VerifiedRoundedIcon from "@mui/icons-material/VerifiedRounded";
import PersonOffRoundedIcon from "@mui/icons-material/PersonOffRounded";
import WarningAmberRoundedIcon from "@mui/icons-material/WarningAmberRounded";
import { formatPercent } from "@/utils/formatters";

export default function AttendanceStats({ summary = {}, activeSession }) {
  const theme = useTheme();

  const cards = [
    {
      label: "Presence Rate",
      value: formatPercent(summary.presenceRate || 0),
      progress: Math.round((summary.presenceRate || 0) * 100),
      icon: HowToRegRoundedIcon,
      tone: "primary"
    },
    {
      label: "Present",
      value: summary.presentCount || 0,
      progress:
        summary.totalStudents && summary.presentCount
          ? Math.round((summary.presentCount / summary.totalStudents) * 100)
          : 0,
      icon: VerifiedRoundedIcon,
      tone: "success"
    },
    {
      label: "Absent",
      value: summary.absenceCount || 0,
      progress:
        summary.totalStudents && summary.absenceCount
          ? Math.round((summary.absenceCount / summary.totalStudents) * 100)
          : 0,
      icon: PersonOffRoundedIcon,
      tone: "warning"
    },
    {
      label: "Unknown or Rejected",
      value: summary.unknownRejected || 0,
      progress: Math.min((summary.unknownRejected || 0) * 10, 100),
      icon: WarningAmberRoundedIcon,
      tone: "secondary"
    }
  ];

  return (
    <Grid container spacing={3}>
      {cards.map((card) => {
        const Icon = card.icon;
        const toneColor = theme.palette[card.tone]?.main || theme.palette.primary.main;

        return (
          <Grid item key={card.label} lg={3} md={6} xs={12}>
            <Card sx={{ height: "100%" }}>
              <CardContent>
                <Stack direction="row" justifyContent="space-between" spacing={2}>
                  <Box>
                    <Typography color="text.secondary" sx={{ letterSpacing: 1 }} variant="caption">
                      Attendance Metric
                    </Typography>
                    <Typography color="text.secondary" sx={{ mt: 0.55 }} variant="body2">
                      {card.label}
                    </Typography>
                    <Typography sx={{ mt: 1.2, lineHeight: 1.1 }} variant="h4">
                      {card.value}
                    </Typography>
                  </Box>
                  <Stack
                    alignItems="center"
                    justifyContent="center"
                    sx={{
                      width: 44,
                      height: 44,
                      borderRadius: 3,
                      bgcolor: alpha(toneColor, 0.14)
                    }}
                  >
                    <Icon fontSize="small" sx={{ color: toneColor }} />
                  </Stack>
                </Stack>
                <LinearProgress
                  sx={{ mt: 2.1, height: 8 }}
                  value={card.progress}
                  variant="determinate"
                />
                <Typography color="text.secondary" sx={{ mt: 1.1 }} variant="caption">
                  {activeSession
                    ? `Active session: ${activeSession.title}`
                    : "No live session in progress"}
                </Typography>
              </CardContent>
            </Card>
          </Grid>
        );
      })}
    </Grid>
  );
}
