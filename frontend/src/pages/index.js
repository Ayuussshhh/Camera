import { useEffect } from "react";
import { CircularProgress, Stack } from "@mui/material";
import { useRouter } from "next/router";
import useAuth from "@/hooks/useAuth";

export default function HomePage() {
  const router = useRouter();
  const { user, loading } = useAuth();

  useEffect(() => {
    if (loading) {
      return;
    }

    router.replace(user ? "/dashboard" : "/login");
  }, [loading, router, user]);

  return (
    <Stack alignItems="center" justifyContent="center" sx={{ minHeight: "100vh" }}>
      <CircularProgress />
    </Stack>
  );
}
