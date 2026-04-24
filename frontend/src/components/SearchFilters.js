import {
  Box,
  Card,
  CardContent,
  Grid,
  MenuItem,
  Stack,
  TextField,
  Typography
} from "@mui/material";
import { SecondaryButton } from "@/components/CustomButtons";

export default function SearchFilters({
  filters,
  fields = [],
  onChange,
  onReset,
  actions
}) {
  return (
    <Card>
      <CardContent sx={{ pb: "24px !important" }}>
        <Stack
          direction={{ md: "row", xs: "column" }}
          spacing={2}
          sx={{ justifyContent: "space-between", alignItems: { md: "center", xs: "stretch" } }}
        >
          <Box>
            <Typography variant="h6">Filter Workspace</Typography>
            <Typography color="text.secondary" sx={{ mt: 0.7 }} variant="body2">
              Refine results by date, department, academic track, and roster parameters.
            </Typography>
          </Box>

          <Stack direction={{ sm: "row", xs: "column" }} spacing={1.25}>
            <SecondaryButton onClick={onReset}>Reset Filters</SecondaryButton>
            {actions}
          </Stack>
        </Stack>

        <Grid alignItems="center" container spacing={2} sx={{ mt: 0.5 }}>
          {fields.map((field) => (
            <Grid item key={field.key} lg={field.lg || 3} md={field.md || 4} sm={6} xs={12}>
              {field.type === "select" ? (
                <TextField
                  fullWidth
                  label={field.label}
                  select
                  value={filters[field.key] || ""}
                  onChange={(event) => onChange(field.key, event.target.value)}
                >
                  <MenuItem value="">All</MenuItem>
                  {field.options?.map((option) => (
                    <MenuItem key={option.value} value={option.value}>
                      {option.label}
                    </MenuItem>
                  ))}
                </TextField>
              ) : (
                <TextField
                  fullWidth
                  label={field.label}
                  placeholder={field.placeholder}
                  type={field.type === "date" ? "date" : "text"}
                  value={filters[field.key] || ""}
                  onChange={(event) => onChange(field.key, event.target.value)}
                  InputLabelProps={field.type === "date" ? { shrink: true } : undefined}
                />
              )}
            </Grid>
          ))}
        </Grid>
      </CardContent>
    </Card>
  );
}
