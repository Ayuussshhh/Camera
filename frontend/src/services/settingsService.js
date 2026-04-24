import apiClient from "@/services/apiClient";

const settingsService = {
  async getSettings() {
    const response = await apiClient.get("/api/settings");
    return response.data.data;
  },
  async updateSettings(payload) {
    const response = await apiClient.put("/api/settings", payload);
    return response.data.data;
  }
};

export default settingsService;
