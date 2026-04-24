import { useEffect } from "react";
import { useRouter } from "next/router";
import useAuth from "@/hooks/useAuth";

export default function useAuthGuard(roles = []) {
  const router = useRouter();
  const { user, loading } = useAuth();

  useEffect(() => {
    if (loading) {
      return;
    }

    if (!user) {
      router.replace("/login");
      return;
    }

    if (roles.length > 0 && !roles.includes(user.role)) {
      router.replace("/dashboard");
    }
  }, [loading, roles, router, user]);

  return {
    user,
    loading,
    authorized:
      Boolean(user) && (roles.length === 0 || roles.includes(user.role))
  };
}
