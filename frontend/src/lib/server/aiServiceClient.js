import { getAiServiceUrl, safeFetchJson } from "@/lib/server/dataAccess";

function normalizeAiPayload(payload) {
  return payload?.data || payload || {};
}

export async function getAiHealth() {
  const aiServiceUrl = getAiServiceUrl();

  try {
    const payload = await safeFetchJson(`${aiServiceUrl}/health`, { method: "GET" }, 5000);

    return {
      connected: true,
      mode: "live",
      url: aiServiceUrl,
      message: payload.message || "Python AI service connected.",
      engine: payload.engine || null,
      recentEvents: payload.recent_events || payload.recentEvents || []
    };
  } catch (error) {
    return {
      connected: false,
      mode: "offline",
      url: aiServiceUrl,
      message: error.message,
      engine: null,
      recentEvents: []
    };
  }
}

export async function registerStudentWithAi({
  studentId,
  studentName,
  images,
  engine
}) {
  const aiServiceUrl = getAiServiceUrl();

  const payload = normalizeAiPayload(
    await safeFetchJson(
      `${aiServiceUrl}/register-student`,
      {
        method: "POST",
        body: JSON.stringify({
          student_id: studentId,
          student_name: studentName,
          images,
          engine
        })
      },
      30000
    )
  );

  const training = payload.training || payload;

  return {
    mode: "live",
    registered: true,
    trained: training.status === "trained",
    message: payload.message || "Student registered and trained in the Python AI service.",
    engine: training.engine || engine,
    imageCount:
      Number(training.image_count || training.imageCount || payload.images_saved || images.length) ||
      images.length,
    embeddingPath: training.embedding_path || training.embeddingPath || null,
    lastTrainedAt:
      training.last_trained_at || training.lastTrainedAt || new Date().toISOString(),
    qualityScore: Number(training.quality_score || training.qualityScore || 0),
    embeddingDimensions: Number(
      training.embedding_dimensions || training.embeddingDimensions || 0
    )
  };
}

export async function startAiSession({
  sessionId,
  engine,
  threshold,
  metadata = {}
}) {
  const aiServiceUrl = getAiServiceUrl();

  return normalizeAiPayload(
    await safeFetchJson(
      `${aiServiceUrl}/start-attendance`,
      {
        method: "POST",
        body: JSON.stringify({
          session_id: sessionId,
          engine,
          threshold,
          metadata
        })
      },
      10000
    )
  );
}

export async function stopAiSession(sessionId) {
  const aiServiceUrl = getAiServiceUrl();

  return normalizeAiPayload(
    await safeFetchJson(
      `${aiServiceUrl}/stop-attendance`,
      {
        method: "POST",
        body: JSON.stringify({
          session_id: sessionId
        })
      },
      10000
    )
  );
}

export async function recognizeFrameWithAi({
  frame,
  sessionId,
  engine,
  threshold
}) {
  const aiServiceUrl = getAiServiceUrl();

  const payload = normalizeAiPayload(
    await safeFetchJson(
      `${aiServiceUrl}/recognize-frame`,
      {
        method: "POST",
        body: JSON.stringify({
          frame,
          session_id: sessionId,
          engine,
          threshold
        })
      },
      20000
    )
  );

  return payload.results || [];
}
