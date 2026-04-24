import ReportsView from "@/views/dashboard/ReportsView";
import getUserLayout from "@/layouts/getUserLayout";

export default function ReportsPage() {
  return <ReportsView />;
}

ReportsPage.getLayout = getUserLayout({
  pageTitle: "Reports & Analytics",
  pageSubtitle: "Analyze attendance trends and export polished institutional reports."
});
