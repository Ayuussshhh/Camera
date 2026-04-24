from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class RegisterStudentRequest(BaseModel):
    student_id: str
    student_name: str
    images: List[str] = Field(default_factory=list)
    engine: str = "mediapipe"


class TrainModelRequest(BaseModel):
    student_id: Optional[str] = None
    engine: str = "mediapipe"


class StartAttendanceRequest(BaseModel):
    session_id: str
    engine: str = "mediapipe"
    threshold: float = 0.91
    metadata: Dict[str, Any] = Field(default_factory=dict)


class StopAttendanceRequest(BaseModel):
    session_id: Optional[str] = None


class RecognizeFrameRequest(BaseModel):
    frame: str
    session_id: Optional[str] = None
    engine: Optional[str] = None
    threshold: Optional[float] = None
