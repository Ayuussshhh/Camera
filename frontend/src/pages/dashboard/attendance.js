import AttendanceView from "@/views/dashboard/AttendanceView";
import getUserLayout from "@/layouts/getUserLayout";

export default function AttendancePage() {
  return <AttendanceView />;
}

AttendancePage.getLayout = getUserLayout({
  pageTitle: "Attendance Records",
  pageSubtitle: "Review session-level logs, daily records, and class-wise attendance outcomes."
});
