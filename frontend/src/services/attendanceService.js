import apiClient from "@/services/apiClient";

const attendanceService = {
  async getRecords(params = {}) {
    const response = await apiClient.get("/api/attendance/records", { params });
    return response.data.data;
  },
  async getSessions() {
    const response = await apiClient.get("/api/attendance/sessions");
    return response.data.data;
  },
  async startSession(payload) {
    const response = await apiClient.post("/api/attendance/sessions", payload);
    return response.data.data;
  },
  async stopSession(sessionId, actor) {
    const response = await apiClient.post("/api/attendance/stop", { sessionId, actor });
    return response.data.data;
  },
  async markManual(payload) {
    const response = await apiClient.post("/api/attendance/manual", payload);
    return response.data.data;
  }
};

export default attendanceService;
