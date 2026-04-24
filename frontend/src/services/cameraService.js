import apiClient from "@/services/apiClient";

const cameraService = {
  async getStatus() {
    const response = await apiClient.get("/api/camera/status");
    return response.data.data;
  },
  async recognizeFrame(payload) {
    const response = await apiClient.post("/api/camera/recognize", payload);
    return response.data.data;
  }
};

export default cameraService;
