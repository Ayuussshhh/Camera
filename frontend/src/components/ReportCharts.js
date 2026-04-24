import {
  Card,
  CardContent,
  Grid,
  Typography
} from "@mui/material";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

function ChartCard({ title, children }) {
  return (
    <Card sx={{ height: "100%" }}>
      <CardContent>
        <Typography sx={{ mb: 2 }} variant="h6">
          {title}
        </Typography>
        {children}
      </CardContent>
    </Card>
  );
}

export default function ReportCharts({ analytics }) {
  return (
    <Grid container spacing={3}>
      <Grid item lg={6} xs={12}>
        <ChartCard title="Daily Attendance Trend">
          <ResponsiveContainer height={280} width="100%">
            <LineChart data={analytics?.dailyTrend || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Legend />
              <Line
                dataKey="present"
                name="Present"
                stroke="#0f766e"
                strokeWidth={3}
                type="monotone"
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </Grid>
      <Grid item lg={6} xs={12}>
        <ChartCard title="Department Breakdown">
          <ResponsiveContainer height={280} width="100%">
            <BarChart data={analytics?.departmentDistribution || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="department" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Legend />
              <Bar dataKey="students" fill="#0f766e" name="Students" radius={[8, 8, 0, 0]} />
              <Bar dataKey="present" fill="#fb923c" name="Present" radius={[8, 8, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>
      </Grid>
      <Grid item xs={12}>
        <ChartCard title="Monthly Attendance Volume">
          <ResponsiveContainer height={280} width="100%">
            <LineChart data={analytics?.monthlyTrend || []}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="date" />
              <YAxis allowDecimals={false} />
              <Tooltip />
              <Legend />
              <Line
                dataKey="present"
                name="Attendance"
                stroke="#ea580c"
                strokeWidth={3}
                type="monotone"
              />
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>
      </Grid>
    </Grid>
  );
}
