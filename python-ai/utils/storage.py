import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import cv2

from utils.config import settings
from utils.engine import get_storage_engine_names, normalize_engine_name
from utils.image import decode_base64_image


def ensure_storage() -> None:
    settings.datasets_dir.mkdir(parents=True, exist_ok=True)
    settings.embeddings_dir.mkdir(parents=True, exist_ok=True)


def save_student_profile(student_id: str, student_name: str, images: List[str]) -> Dict[str, object]:
    ensure_storage()
    student_dir = settings.datasets_dir / student_id
    student_dir.mkdir(parents=True, exist_ok=True)

    for image_path in student_dir.glob("*.jpg"):
        image_path.unlink(missing_ok=True)

    metadata = {
        "student_id": student_id,
        "student_name": student_name,
        "image_count": len(images),
        "updated_at": datetime.utcnow().isoformat(),
    }

    for index, image_data in enumerate(images):
        image = decode_base64_image(image_data)
        image_path = student_dir / f"face_{index + 1}.jpg"
        cv2.imwrite(str(image_path), image)

    with open(student_dir / "metadata.json", "w", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    return metadata


def load_student_profile(student_id: str) -> Dict[str, object]:
    student_dir = settings.datasets_dir / student_id
    metadata_path = student_dir / "metadata.json"

    if not metadata_path.exists():
        raise ValueError(f"Student profile '{student_id}' not found.")

    with open(metadata_path, "r", encoding="utf-8") as metadata_file:
        return json.load(metadata_file)


def list_student_profiles() -> List[Dict[str, object]]:
    ensure_storage()
    profiles = []

    for student_dir in settings.datasets_dir.iterdir():
        if not student_dir.is_dir():
            continue

        metadata_path = student_dir / "metadata.json"

        if metadata_path.exists():
            with open(metadata_path, "r", encoding="utf-8") as metadata_file:
                profiles.append(json.load(metadata_file))

    return profiles


def load_student_images(student_id: str) -> List[object]:
    student_dir = settings.datasets_dir / student_id

    if not student_dir.exists():
        raise ValueError(f"Student dataset '{student_id}' not found.")

    images = []

    for image_path in sorted(student_dir.glob("*.jpg")):
        image = cv2.imread(str(image_path))

        if image is not None:
            images.append(image)

    return images


def _raw_embedding_path(directory_name: str) -> Path:
    ensure_storage()
    engine_dir = settings.embeddings_dir / directory_name
    engine_dir.mkdir(parents=True, exist_ok=True)
    return engine_dir


def _embedding_path(engine_name: str) -> Path:
    return _raw_embedding_path(normalize_engine_name(engine_name))


def _student_embedding_path(engine_name: str, student_id: str) -> Path:
    return _embedding_path(engine_name) / f"{student_id}.json"


def _gallery_path(engine_name: str) -> Path:
    return _embedding_path(engine_name) / "gallery.json"


def build_embedding_path(engine_name: str, student_id: str) -> str:
    return _student_embedding_path(engine_name, student_id).relative_to(
        settings.base_dir
    ).as_posix()


def _load_records_from_directory(engine_dir: Path) -> List[Dict[str, object]]:
    gallery_path = engine_dir / "gallery.json"

    if gallery_path.exists():
        with open(gallery_path, "r", encoding="utf-8") as embedding_file:
            return json.load(embedding_file)

    records = []

    for record_path in sorted(engine_dir.glob("*.json")):
        if record_path.name == "gallery.json":
            continue

        with open(record_path, "r", encoding="utf-8") as embedding_file:
            records.append(json.load(embedding_file))

    return records


def load_embedding_records(engine_name: str) -> List[Dict[str, object]]:
    records_by_student: Dict[str, Dict[str, object]] = {}

    for directory_name in reversed(get_storage_engine_names(engine_name)):
        engine_dir = _raw_embedding_path(directory_name)

        for record in _load_records_from_directory(engine_dir):
            student_id = str(record.get("student_id") or "")

            if not student_id:
                continue

            records_by_student[student_id] = {
                **record,
                "engine": normalize_engine_name(record.get("engine")),
            }

    return list(records_by_student.values())


def upsert_embedding_record(engine_name: str, payload: Dict[str, object]) -> Dict[str, object]:
    record_path = _student_embedding_path(engine_name, payload["student_id"])
    updated_payload = {
        **payload,
        "engine": normalize_engine_name(payload.get("engine")),
        "updated_at": datetime.utcnow().isoformat(),
    }

    with open(record_path, "w", encoding="utf-8") as embedding_file:
        json.dump(updated_payload, embedding_file, indent=2)

    records = load_embedding_records(engine_name)

    with open(_gallery_path(engine_name), "w", encoding="utf-8") as gallery_file:
        json.dump(records, gallery_file, indent=2)

    return updated_payload
