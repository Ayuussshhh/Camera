from collections import deque
import time
import uuid
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np

from engines import get_engine
from models.schemas import RecognizeFrameRequest
from services.anti_spoof_service import anti_spoof_service
from services.attendance_service import attendance_service
from utils.config import settings
from utils.engine import normalize_engine_name
from utils.image import decode_base64_image
from utils.storage import load_embedding_records


def cosine_similarity(vector_a: List[float], vector_b: List[float]) -> float:
    first = np.asarray(vector_a, dtype=np.float32)
    second = np.asarray(vector_b, dtype=np.float32)

    if first.size == 0 or second.size == 0 or first.shape != second.shape:
        return 0.0

    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))

    if denominator == 0.0:
        return 0.0

    return float(np.dot(first, second) / denominator)


class RecognitionService:
    def __init__(self) -> None:
        self._session_history: Dict[str, Deque[Dict[str, object]]] = {}
        self._recent_events: Deque[Dict[str, object]] = deque(maxlen=100)

    def recognize_frame(self, payload: RecognizeFrameRequest) -> Dict[str, object]:
        session_state = attendance_service.get_state()
        engine_name = normalize_engine_name(
            payload.engine or session_state["engine"] or settings.default_engine
        )
        threshold = (
            payload.threshold
            if payload.threshold is not None
            else session_state["threshold"] or settings.default_threshold
        )

        frame = decode_base64_image(payload.frame)
        engine = get_engine(engine_name)
        detections = engine.extract_embeddings(frame)
        gallery = load_embedding_records(engine_name)

        results = []
        session_id = payload.session_id or session_state["session_id"] or "standalone"

        for detection in detections:
            anti_spoof = anti_spoof_service.evaluate(frame, detection["bbox"])
            match, confidence = self._find_best_match(detection["embedding"], gallery)
            smoothed_confidence = self._smooth_confidence(
                session_id,
                match["student_id"] if match else None,
                confidence,
            )
            quality = float(detection.get("quality", 0.0))

            if settings.liveness_enabled and not anti_spoof["passed"]:
                result = {
                    "student_id": None,
                    "label": "Spoof Rejected",
                    "confidence": smoothed_confidence,
                    "bbox": detection["bbox"],
                    "anti_spoof": anti_spoof,
                    "quality": round(quality, 4),
                }
                self._record_event(session_id, result)
                results.append(result)
                continue

            if quality < 0.22:
                result = {
                    "student_id": None,
                    "label": "Low Quality Face",
                    "confidence": smoothed_confidence,
                    "bbox": detection["bbox"],
                    "anti_spoof": anti_spoof,
                    "quality": round(quality, 4),
                }
                self._record_event(session_id, result)
                results.append(result)
                continue

            if match and smoothed_confidence >= threshold:
                result = {
                    "student_id": match["student_id"],
                    "label": match["student_name"],
                    "confidence": smoothed_confidence,
                    "bbox": detection["bbox"],
                    "anti_spoof": anti_spoof,
                    "quality": round(quality, 4),
                    "engine": match.get("engine", engine_name),
                }
            else:
                result = {
                    "student_id": None,
                    "label": "Unknown",
                    "confidence": smoothed_confidence,
                    "bbox": detection["bbox"],
                    "anti_spoof": anti_spoof,
                    "quality": round(quality, 4),
                }

            self._record_event(session_id, result)
            results.append(result)

        return {
            "status": "processed",
            "session_id": payload.session_id or session_state["session_id"],
            "engine": engine_name,
            "threshold": threshold,
            "max_faces_per_frame": 5,
            "detected_faces": len(results),
            "results": results,
            "recent_events": self.get_recent_events(limit=15),
        }

    def get_recent_events(self, limit: int = 20) -> List[Dict[str, object]]:
        return list(self._recent_events)[:limit]

    def clear_session_history(self, session_id: Optional[str]) -> None:
        if not session_id:
            return

        self._session_history.pop(session_id, None)

    def _record_event(self, session_id: str, result: Dict[str, object]) -> None:
        self._recent_events.appendleft(
            {
                "event_id": str(uuid.uuid4()),
                "session_id": session_id,
                "student_id": result.get("student_id"),
                "label": result.get("label"),
                "confidence": round(float(result.get("confidence", 0.0)), 4),
                "quality": round(float(result.get("quality", 0.0)), 4),
                "timestamp": time.time(),
            }
        )

    def _get_session_history(self, session_id: str) -> Deque[Dict[str, object]]:
        if session_id not in self._session_history:
            self._session_history[session_id] = deque(maxlen=40)

        return self._session_history[session_id]

    def _smooth_confidence(
        self,
        session_id: str,
        student_id: Optional[str],
        confidence: float,
    ) -> float:
        history = self._get_session_history(session_id)
        current_time = time.time()
        recent_history = [
            float(item["confidence"])
            for item in history
            if item.get("student_id") == student_id
            and (current_time - float(item["timestamp"])) <= 10
        ]

        blended_confidence = (
            ((confidence * 2.0) + sum(recent_history)) / (2 + len(recent_history))
            if recent_history
            else confidence
        )

        history.append(
            {
                "student_id": student_id,
                "confidence": blended_confidence,
                "timestamp": current_time,
            }
        )

        return round(blended_confidence, 4)

    def _find_best_match(
        self,
        embedding: List[float],
        gallery: List[Dict[str, object]],
    ) -> Tuple[Optional[Dict[str, object]], float]:
        best_match = None
        best_score = 0.0

        for candidate in gallery:
            profile_score = cosine_similarity(embedding, candidate.get("embedding", []))
            sample_embeddings = candidate.get("sample_embeddings") or []
            sample_scores = sorted(
                (
                    cosine_similarity(embedding, sample_embedding)
                    for sample_embedding in sample_embeddings
                ),
                reverse=True,
            )
            sample_score = (
                sum(sample_scores[:3]) / min(len(sample_scores), 3)
                if sample_scores
                else profile_score
            )
            score = (0.68 * profile_score) + (0.32 * sample_score)

            if score > best_score:
                best_score = score
                best_match = candidate

        return best_match, round(best_score, 4)


recognition_service = RecognitionService()
