export const CANONICAL_ENGINE = "mediapipe";

const ENGINE_ALIASES = new Set([
  CANONICAL_ENGINE,
  "opencv",
  "face_recognition",
  "deepface",
  "insightface",
  "arcface"
]);

export function normalizeEngineName(engine) {
  const value = String(engine || "")
    .trim()
    .toLowerCase();

  if (!value || ENGINE_ALIASES.has(value)) {
    return CANONICAL_ENGINE;
  }

  return CANONICAL_ENGINE;
}

export function getEngineLabel(engine) {
  return normalizeEngineName(engine) === CANONICAL_ENGINE
    ? "MediaPipe Face Embeddings"
    : "MediaPipe Face Embeddings";
}
