import DashboardOverviewView from "@/views/dashboard/DashboardOverviewView";
import getUserLayout from "@/layouts/getUserLayout";

export default function DashboardPage() {
  return <DashboardOverviewView />;
}

DashboardPage.getLayout = getUserLayout({
  pageTitle: "Dashboard Overview",
  pageSubtitle: "Track attendance health, session throughput, and recognition system status."
});
