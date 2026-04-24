import SettingsView from "@/views/dashboard/SettingsView";
import getUserLayout from "@/layouts/getUserLayout";

export default function SettingsPage() {
  return <SettingsView />;
}

SettingsPage.getLayout = getUserLayout({
  roles: ["admin"],
  pageTitle: "Platform Settings",
  pageSubtitle: "Configure recognition behavior, controls, and platform-level defaults."
});
