import apiClient from "@/services/apiClient";

const authService = {
  async login(payload) {
    const response = await apiClient.post("/api/auth/login", payload);
    return response.data.data;
  }
};

export default authService;
