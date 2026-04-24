import argparse
import sys
from pathlib import Path
from typing import Iterable, List
from urllib.request import urlretrieve

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from engines import get_engine


DEFAULT_SAME_IDENTITY = [
    "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama.jpg",
    "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama2.jpg",
    "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/obama_small.jpg",
]
DEFAULT_DIFFERENT_IDENTITY = [
    "https://raw.githubusercontent.com/ageitgey/face_recognition/master/examples/biden.jpg",
]


def cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    first = np.asarray(vector_a, dtype=np.float32)
    second = np.asarray(vector_b, dtype=np.float32)
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))

    if denominator == 0.0:
        return 0.0

    return float(np.dot(first, second) / denominator)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the MediaPipe embedding pipeline with sample or user-supplied images."
    )
    parser.add_argument(
        "--same",
        nargs="+",
        default=DEFAULT_SAME_IDENTITY,
        help="Two or more image paths or URLs for the same identity.",
    )
    parser.add_argument(
        "--different",
        nargs="+",
        default=DEFAULT_DIFFERENT_IDENTITY,
        help="One or more image paths or URLs for a different identity.",
    )
    parser.add_argument(
        "--cache-dir",
        default=str(Path(__file__).resolve().parent.parent / "verification-cache"),
        help="Directory used for downloaded sample images.",
    )
    return parser.parse_args()


def resolve_image(source: str, cache_dir: Path) -> Path:
    if source.startswith("http://") or source.startswith("https://"):
        cache_dir.mkdir(parents=True, exist_ok=True)
        destination = cache_dir / Path(source).name

        if not destination.exists():
            urlretrieve(source, destination)

        return destination

    return Path(source).expanduser().resolve()


def load_images(sources: Iterable[str], cache_dir: Path) -> List[np.ndarray]:
    images: List[np.ndarray] = []

    for source in sources:
        image_path = resolve_image(source, cache_dir)
        image = cv2.imread(str(image_path))

        if image is None:
            raise RuntimeError(f"Unable to read image '{image_path.as_posix()}'.")

        images.append(image)

    return images


def extract_best_detection(engine, image: np.ndarray, label: str) -> dict:
    detections = engine.extract_embeddings(image)

    if not detections:
        raise RuntimeError(f"No face detected in '{label}'.")

    return max(detections, key=lambda item: float(item.get("quality", 0.0)))


def main() -> int:
    args = parse_args()

    if len(args.same) < 2:
        raise RuntimeError("Provide at least two '--same' images.")

    cache_dir = Path(args.cache_dir).resolve()
    engine = get_engine("mediapipe")
    same_images = load_images(args.same, cache_dir)
    different_images = load_images(args.different, cache_dir)

    profile = engine.compute_student_profile(same_images[:-1] or same_images[:1])
    same_probe = extract_best_detection(engine, same_images[-1], "same-identity probe")
    different_probe = extract_best_detection(
        engine,
        different_images[0],
        "different-identity probe",
    )

    same_score = cosine_similarity(profile["embedding"], same_probe["embedding"])
    different_score = cosine_similarity(profile["embedding"], different_probe["embedding"])
    score_gap = same_score - different_score
    suggested_threshold = round((same_score + different_score) / 2.0, 4)

    print("MediaPipe embedding verification")
    print(f"Accepted enrollment samples : {profile['accepted_image_count']}")
    print(f"Rejected enrollment samples : {profile['rejected_image_count']}")
    print(f"Profile quality score       : {profile['quality_score']:.4f}")
    print(f"Probe quality (same)        : {same_probe['quality']:.4f}")
    print(f"Probe quality (different)   : {different_probe['quality']:.4f}")
    print(f"Similarity (same identity)  : {same_score:.4f}")
    print(f"Similarity (different)      : {different_score:.4f}")
    print(f"Similarity gap              : {score_gap:.4f}")
    print(f"Suggested threshold         : {suggested_threshold:.4f}")

    if same_score <= different_score:
        print("Verification result         : FAILED")
        return 1

    print("Verification result         : PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
