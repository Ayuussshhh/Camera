import apiClient from "@/services/apiClient";

const studentService = {
  async getStudents(params = {}) {
    const response = await apiClient.get("/api/students", { params });
    return response.data.data;
  },
  async createStudent(payload) {
    const response = await apiClient.post("/api/students", payload);
    return response.data.data;
  },
  async getStudent(studentId) {
    const response = await apiClient.get(`/api/students/${studentId}`);
    return response.data.data;
  },
  async updateStudent(studentId, payload) {
    const response = await apiClient.put(`/api/students/${studentId}`, payload);
    return response.data.data;
  },
  async enrollFace(payload) {
    const response = await apiClient.post("/api/students/enroll", payload);
    return response.data.data;
  }
};

export default studentService;
