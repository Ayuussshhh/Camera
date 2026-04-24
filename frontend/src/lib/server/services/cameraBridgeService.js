import { getActiveSession, markAttendance } from "@/lib/server/services/attendanceService";
import { getStudentById } from "@/lib/server/services/studentService";
import {
  getAiHealth,
  recognizeFrameWithAi,
  registerStudentWithAi as registerWithAi
} from "@/lib/server/aiServiceClient";

function mapExternalResult(result) {
  return {
    studentId: result.student_id || result.studentId || null,
    confidence: Number(result.confidence || 0),
    label: result.label || result.name || "Unknown",
    bbox: result.bbox || null,
    antiSpoof: result.anti_spoof || result.antiSpoof || null,
    quality: Number(result.quality || 0)
  };
}

function dedupeRecognitions(recognitions) {
  const bestByStudent = new Map();
  const passthrough = [];

  for (const recognition of recognitions) {
    if (!recognition.studentId) {
      passthrough.push(recognition);
      continue;
    }

    const existing = bestByStudent.get(recognition.studentId);

    if (!existing || recognition.confidence > existing.confidence) {
      bestByStudent.set(recognition.studentId, recognition);
    }
  }

  return [...bestByStudent.values(), ...passthrough].sort(
    (first, second) => Number(second.confidence || 0) - Number(first.confidence || 0)
  );
}

export { getAiHealth };

export async function registerStudentWithAi({ studentId, images, engine }) {
  const student = await getStudentById(studentId);
  return registerWithAi({
    studentId: student.id,
    studentName: student.fullName,
    images,
    engine
  });
}

export async function recognizeFrame(payload, actor = {}) {
  const session = await getActiveSession();

  if (!session) {
    throw new Error("No active attendance session is available.");
  }

  const aiHealth = await getAiHealth();

  if (!aiHealth.connected) {
    throw new Error(
      aiHealth.message || "Python AI service is unavailable. Recognition cannot continue."
    );
  }

  const recognitions = dedupeRecognitions(
    (
    await recognizeFrameWithAi({
      frame: payload.frame,
      sessionId: session.id,
      engine: payload.engine || session.engine,
      threshold: payload.threshold
    })
    ).map(mapExternalResult)
  );

  const results = [];

  for (const recognition of recognitions) {
    if (
      recognition.label === "Low Quality Face" ||
      recognition.label === "Spoof Rejected"
    ) {
      results.push({
        ...recognition,
        attendanceStatus: "skipped",
        message:
          recognition.label === "Low Quality Face"
            ? "Capture skipped because the detected face quality is too low."
            : "Capture rejected by anti-spoof validation.",
        attendance: null
      });
      continue;
    }

    const attendanceResult = await markAttendance(
      {
        sessionId: session.id,
        studentId: recognition.studentId,
        confidence: recognition.confidence,
        source: `live-${payload.engine || session.engine}`
      },
      actor
    );

    results.push({
      ...recognition,
      attendanceStatus: attendanceResult.status,
      message: attendanceResult.message,
      attendance: attendanceResult.attendance || null
    });
  }

  return {
    sessionId: session.id,
    sourceMode: "live",
    aiHealth,
    maxFacesPerFrame: 5,
    results
  };
}
