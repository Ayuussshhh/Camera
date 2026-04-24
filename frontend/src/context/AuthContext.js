import { createContext, useContext, useEffect, useState } from "react";
import Router from "next/router";
import authService from "@/services/authService";

const AuthContext = createContext({
  user: null,
  loading: true,
  login: async () => {},
  logout: () => {}
});

const STORAGE_KEY = "attendance-auth-user";

export function AuthProvider({ children }) {
  const [user, setUser] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    if (typeof window === "undefined") {
      return;
    }

    const storedUser = window.localStorage.getItem(STORAGE_KEY);

    if (storedUser) {
      setUser(JSON.parse(storedUser));
    }

    setLoading(false);
  }, []);

  const login = async (credentials) => {
    const authenticatedUser = await authService.login(credentials);

    setUser(authenticatedUser);

    if (typeof window !== "undefined") {
      window.localStorage.setItem(
        STORAGE_KEY,
        JSON.stringify(authenticatedUser)
      );
    }

    return authenticatedUser;
  };

  const logout = () => {
    setUser(null);

    if (typeof window !== "undefined") {
      window.localStorage.removeItem(STORAGE_KEY);
    }

    Router.push("/login");
  };

  return (
    <AuthContext.Provider value={{ user, loading, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
