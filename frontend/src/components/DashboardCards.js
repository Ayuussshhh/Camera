import {
  alpha,
  useTheme
} from "@mui/material/styles";
import {
  Box,
  Card,
  CardContent,
  Grid,
  Stack,
  Typography
} from "@mui/material";
import SchoolRoundedIcon from "@mui/icons-material/SchoolRounded";
import CheckCircleRoundedIcon from "@mui/icons-material/CheckCircleRounded";
import VideocamRoundedIcon from "@mui/icons-material/VideocamRounded";
import ShieldRoundedIcon from "@mui/icons-material/ShieldRounded";

const toneMap = {
  primary: SchoolRoundedIcon,
  success: CheckCircleRoundedIcon,
  secondary: VideocamRoundedIcon,
  warning: ShieldRoundedIcon
};

export default function DashboardCards({ cards = [] }) {
  const theme = useTheme();

  return (
    <Grid container spacing={3}>
      {cards.map((card) => {
        const Icon = toneMap[card.tone] || SchoolRoundedIcon;

        return (
          <Grid item key={card.title} lg={3} md={6} xs={12}>
            <Card
              sx={{
                height: "100%",
                position: "relative",
                overflow: "hidden"
              }}
            >
              <Box
                sx={{
                  position: "absolute",
                  inset: 0,
                  background: `linear-gradient(140deg, ${alpha(
                    theme.palette[card.tone]?.main || theme.palette.primary.main,
                    0.08
                  )} 0%, transparent 52%)`,
                  pointerEvents: "none"
                }}
              />
              <CardContent>
                <Stack direction="row" justifyContent="space-between" spacing={2.2}>
                  <Box sx={{ minWidth: 0 }}>
                    <Typography color="text.secondary" sx={{ letterSpacing: 1 }} variant="caption">
                      Operations
                    </Typography>
                    <Typography color="text.secondary" sx={{ mt: 0.6 }} variant="body2">
                      {card.title}
                    </Typography>
                    <Typography sx={{ mt: 1.1, lineHeight: 1.1 }} variant="h4">
                      {card.value}
                    </Typography>
                    <Typography color="text.secondary" sx={{ mt: 1.1 }} variant="body2">
                      {card.subtitle}
                    </Typography>
                  </Box>

                  <Stack
                    alignItems="center"
                    justifyContent="center"
                    sx={{
                      width: 56,
                      height: 56,
                      borderRadius: 2,
                      bgcolor: alpha(theme.palette[card.tone]?.main || theme.palette.primary.main, 0.12)
                    }}
                  >
                    <Icon color={card.tone} />
                  </Stack>
                </Stack>
              </CardContent>
            </Card>
          </Grid>
        );
      })}
    </Grid>
  );
}
