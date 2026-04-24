import os
from pathlib import Path

from utils.engine import normalize_engine_name


class Settings:
    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        self.base_dir = base_dir
        self.datasets_dir = base_dir / "datasets"
        self.embeddings_dir = base_dir / "embeddings"
        self.models_dir = base_dir / "models"
        self.face_landmarker_model_path = self.models_dir / "face_landmarker.task"
        self.default_engine = normalize_engine_name(
            os.getenv("AI_DEFAULT_ENGINE", "mediapipe")
        )
        self.default_threshold = float(os.getenv("AI_DEFAULT_THRESHOLD", "0.91"))
        self.camera_index = int(os.getenv("AI_CAMERA_INDEX", "0"))
        self.open_camera_on_session_start = (
            os.getenv("AI_OPEN_CAMERA_ON_START", "false").lower() == "true"
        )
        self.liveness_enabled = os.getenv("AI_LIVENESS_ENABLED", "false").lower() == "true"
        self.allowed_origins = os.getenv("AI_ALLOWED_ORIGINS", "*").split(",")


settings = Settings()
