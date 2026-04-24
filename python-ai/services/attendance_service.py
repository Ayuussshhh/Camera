from datetime import datetime
from threading import Lock
from typing import Any, Dict, Optional

from models.schemas import StartAttendanceRequest
from utils.camera import camera_manager
from utils.config import settings
from utils.engine import normalize_engine_name


class AttendanceService:
    def __init__(self) -> None:
        self._lock = Lock()
        self._state: Dict[str, Any] = {
            "active": False,
            "session_id": None,
            "engine": settings.default_engine,
            "threshold": settings.default_threshold,
            "started_at": None,
            "metadata": {},
        }

    def start_session(self, payload: StartAttendanceRequest) -> Dict[str, Any]:
        engine_name = normalize_engine_name(payload.engine or settings.default_engine)

        with self._lock:
            self._state = {
                "active": True,
                "session_id": payload.session_id,
                "engine": engine_name,
                "threshold": payload.threshold,
                "started_at": datetime.utcnow().isoformat(),
                "metadata": payload.metadata,
            }

        if settings.open_camera_on_session_start:
            try:
                camera_manager.open()
            except Exception:
                pass

        return {
            "status": "started",
            "session": self.get_state(),
        }

    def stop_session(self, session_id: Optional[str] = None) -> Dict[str, Any]:
        with self._lock:
            if session_id and self._state["session_id"] != session_id:
                raise ValueError("Session ID does not match the active attendance session.")

            previous_state = dict(self._state)
            self._state = {
                "active": False,
                "session_id": None,
                "engine": settings.default_engine,
                "threshold": settings.default_threshold,
                "started_at": None,
                "metadata": {},
            }

        camera_manager.release()
        return {
            "status": "stopped",
            "session": previous_state,
        }

    def get_state(self) -> Dict[str, Any]:
        return dict(self._state)


attendance_service = AttendanceService()
