from typing import Dict, List

from engines import get_engine
from models.schemas import RegisterStudentRequest, TrainModelRequest
from utils.config import settings
from utils.engine import normalize_engine_name
from utils.storage import (
    build_embedding_path,
    list_student_profiles,
    load_student_images,
    load_student_profile,
    save_student_profile,
    upsert_embedding_record,
)


class EnrollmentService:
    def register_student(self, payload: RegisterStudentRequest) -> Dict[str, object]:
        if not payload.images:
            raise ValueError("At least one image is required for student registration.")

        save_student_profile(payload.student_id, payload.student_name, payload.images)
        training_result = self._train_single_student(
            student_id=payload.student_id,
            student_name=payload.student_name,
            engine_name=payload.engine,
        )

        return {
            "status": "registered",
            "student_id": payload.student_id,
            "student_name": payload.student_name,
            "images_saved": len(payload.images),
            "recommended_image_count": 5,
            "minimum_image_count": 3,
            "training": training_result,
        }

    def train_model(self, payload: TrainModelRequest) -> Dict[str, object]:
        engine_name = normalize_engine_name(payload.engine or settings.default_engine)

        if payload.student_id:
            profile = load_student_profile(payload.student_id)
            return self._train_single_student(
                student_id=profile["student_id"],
                student_name=profile["student_name"],
                engine_name=engine_name,
            )

        profiles = list_student_profiles()
        results: List[Dict[str, object]] = []

        for profile in profiles:
            try:
                results.append(
                    self._train_single_student(
                        student_id=profile["student_id"],
                        student_name=profile["student_name"],
                        engine_name=engine_name,
                    )
                )
            except Exception as error:
                results.append(
                    {
                        "student_id": profile["student_id"],
                        "status": "failed",
                        "message": str(error),
                    }
                )

        return {
            "status": "completed",
            "engine": engine_name,
            "results": results,
        }

    def _train_single_student(
        self,
        student_id: str,
        student_name: str,
        engine_name: str,
    ) -> Dict[str, object]:
        engine_name = normalize_engine_name(engine_name)
        engine = get_engine(engine_name)
        images = load_student_images(student_id)

        if not images:
            raise ValueError(f"No dataset images found for student {student_id}.")

        profile = engine.compute_student_profile(images)
        embedding_path = build_embedding_path(engine_name, student_id)
        embedding_record = upsert_embedding_record(
            engine_name,
            {
                "student_id": student_id,
                "student_name": student_name,
                "engine": engine_name,
                "embedding": profile["embedding"],
                "embedding_path": embedding_path,
                "image_count": profile["accepted_image_count"],
                "quality_score": round(profile["quality_score"], 4),
                "embedding_dimensions": profile["embedding_dimensions"],
                "sample_embeddings": profile["sample_embeddings"],
                "sample_qualities": profile["sample_qualities"],
            },
        )

        return {
            "student_id": student_id,
            "student_name": student_name,
            "engine": engine_name,
            "status": "trained",
            "image_count": profile["accepted_image_count"],
            "source_image_count": profile["image_count"],
            "rejected_image_count": profile["rejected_image_count"],
            "quality_score": round(profile["quality_score"], 4),
            "embedding_dimensions": profile["embedding_dimensions"],
            "sample_count": len(profile["sample_embeddings"]),
            "sample_qualities": profile["sample_qualities"],
            "embedding_path": embedding_record["embedding_path"],
            "last_trained_at": embedding_record["updated_at"],
            "recommended_image_count": 5,
            "minimum_image_count": 3,
        }


enrollment_service = EnrollmentService()
