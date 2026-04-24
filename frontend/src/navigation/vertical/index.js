import DashboardRoundedIcon from "@mui/icons-material/DashboardRounded";
import GroupsRoundedIcon from "@mui/icons-material/GroupsRounded";
import FactCheckRoundedIcon from "@mui/icons-material/FactCheckRounded";
import VideocamRoundedIcon from "@mui/icons-material/VideocamRounded";
import InsightsRoundedIcon from "@mui/icons-material/InsightsRounded";
import SettingsRoundedIcon from "@mui/icons-material/SettingsRounded";

const navigation = () => [
  {
    sectionTitle: "Dashboard"
  },
  {
    title: "Overview",
    icon: DashboardRoundedIcon,
    path: "/dashboard",
    caption: "Summary and system health"
  },
  {
    sectionTitle: "Management"
  },
  {
    title: "Students",
    icon: GroupsRoundedIcon,
    path: "/dashboard/students",
    caption: "Enrollment and profiles"
  },
  {
    title: "Attendance",
    icon: FactCheckRoundedIcon,
    path: "/dashboard/attendance",
    caption: "Records and sessions"
  },
  {
    title: "Live Camera",
    icon: VideocamRoundedIcon,
    path: "/dashboard/live-camera",
    caption: "Real-time recognition",
    badgeContent: "LIVE",
    badgeColor: "success"
  },
  {
    sectionTitle: "Analytics"
  },
  {
    title: "Reports",
    icon: InsightsRoundedIcon,
    path: "/dashboard/reports",
    caption: "Insights and exports"
  },
  {
    sectionTitle: "Settings"
  },
  {
    title: "Settings",
    icon: SettingsRoundedIcon,
    path: "/dashboard/settings",
    caption: "Engine and platform config"
  }
];

export default navigation;
