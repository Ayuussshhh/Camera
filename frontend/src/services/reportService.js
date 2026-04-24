import apiClient from "@/services/apiClient";

const reportService = {
  async getAnalytics(params = {}) {
    const response = await apiClient.get("/api/reports/analytics", { params });
    return response.data.data;
  }
};

export default reportService;
