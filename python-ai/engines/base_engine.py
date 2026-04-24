from abc import ABC, abstractmethod
from typing import Dict, List

import numpy as np


class BaseRecognitionEngine(ABC):
    name = "base"
    minimum_enrollment_quality = 0.3

    @abstractmethod
    def extract_embeddings(self, image: np.ndarray) -> List[Dict[str, object]]:
        raise NotImplementedError

    def compute_student_profile(self, images: List[np.ndarray]) -> Dict[str, object]:
        collected: List[Dict[str, object]] = []
        rejected_image_count = 0

        for image in images:
            detections = self.extract_embeddings(image)

            if not detections:
                rejected_image_count += 1
                continue

            best_detection = max(
                detections,
                key=lambda item: float(item.get("quality", 0.0)),
            )

            if (
                float(best_detection.get("quality", 0.0))
                < self.minimum_enrollment_quality
            ):
                rejected_image_count += 1
                continue

            collected.append(best_detection)

        if not collected:
            raise ValueError(
                f"No high-quality faces detected while training with engine '{self.name}'."
            )

        embeddings = np.asarray(
            [item["embedding"] for item in collected],
            dtype=np.float32,
        )
        weights = np.asarray(
            [max(float(item.get("quality", 0.0)), 0.1) for item in collected],
            dtype=np.float32,
        )
        sorted_collected = sorted(
            collected,
            key=lambda item: float(item.get("quality", 0.0)),
            reverse=True,
        )

        profile_embedding = np.average(embeddings, axis=0, weights=weights)
        profile_embedding = profile_embedding / (
            np.linalg.norm(profile_embedding) + 1e-8
        )

        return {
            "embedding": profile_embedding.astype(np.float32).tolist(),
            "image_count": len(images),
            "accepted_image_count": len(collected),
            "rejected_image_count": rejected_image_count,
            "quality_score": float(np.mean(weights)),
            "embedding_dimensions": int(profile_embedding.shape[0]),
            "sample_embeddings": [
                np.asarray(item["embedding"], dtype=np.float32).tolist()
                for item in sorted_collected[:6]
            ],
            "sample_qualities": [
                round(float(item.get("quality", 0.0)), 4)
                for item in sorted_collected[:6]
            ],
        }

    def compute_student_embedding(self, images: List[np.ndarray]) -> List[float]:
        return self.compute_student_profile(images)["embedding"]
