import StudentsView from "@/views/dashboard/StudentsView";
import getUserLayout from "@/layouts/getUserLayout";

export default function StudentsPage() {
  return <StudentsView />;
}

StudentsPage.getLayout = getUserLayout({
  pageTitle: "Student Management",
  pageSubtitle: "Register students, manage face profiles, and maintain roster quality."
});
