import { useMemo } from "react";
import Layout from "@/@core/layouts/Layout";
import useAuthGuard from "@/hooks/useAuthGuard";
import navigation from "@/navigation/vertical";
import AppBarContent from "@/layouts/components/vertical/AppBarContent";

export default function UserLayout({
  children,
  pageTitle,
  pageSubtitle,
  pageActions,
  roles = []
}) {
  const { loading, authorized } = useAuthGuard(roles);
  const navigationItems = useMemo(() => navigation(), []);

  if (loading || !authorized) {
    return null;
  }

  return (
    <Layout
      appBarContent={(layoutProps) => (
        <AppBarContent
          {...layoutProps}
          actions={pageActions}
          subtitle={pageSubtitle}
          title={pageTitle}
        />
      )}
      navigationItems={navigationItems}
    >
      {children}
    </Layout>
  );
}
