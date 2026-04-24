export const USER_ROLES = [
  { label: "Admin", value: "admin" },
  { label: "Teacher", value: "teacher" }
];

export const SEMESTER_OPTIONS = [
  "1",
  "2",
  "3",
  "4",
  "5",
  "6",
  "7",
  "8"
];

export const SECTION_OPTIONS = ["A", "B", "C", "D"];

export const ENGINE_OPTIONS = [
  { label: "MediaPipe Face Embeddings", value: "mediapipe" }
];

export const DEFAULT_LOGIN_HINTS = [
  {
    role: "admin",
    email: "admin@college.edu",
    password: "admin123"
  },
  {
    role: "teacher",
    email: "teacher@college.edu",
    password: "teacher123"
  }
];
