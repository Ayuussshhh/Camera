import LiveCameraView from "@/views/dashboard/LiveCameraView";
import getUserLayout from "@/layouts/getUserLayout";

export default function LiveCameraPage() {
  return <LiveCameraView />;
}

LiveCameraPage.getLayout = getUserLayout({
  roles: ["admin", "teacher"],
  pageTitle: "Live Camera Session",
  pageSubtitle: "Run real-time attendance sessions with frame scans and instant recognitions."
});
