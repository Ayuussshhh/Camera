from typing import List


CANONICAL_ENGINE = "mediapipe"
ENGINE_ALIASES = {
    CANONICAL_ENGINE,
    "opencv",
    "face_recognition",
    "deepface",
    "insightface",
    "arcface",
}


def normalize_engine_name(name: str | None) -> str:
    value = (name or "").strip().lower()

    if not value or value in ENGINE_ALIASES:
        return CANONICAL_ENGINE

    return CANONICAL_ENGINE


def get_storage_engine_names(name: str | None) -> List[str]:
    normalized_name = normalize_engine_name(name)

    if normalized_name == CANONICAL_ENGINE:
        return [CANONICAL_ENGINE, "opencv"]

    return [normalized_name]
