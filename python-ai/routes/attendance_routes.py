from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from models.schemas import (
    RecognizeFrameRequest,
    RegisterStudentRequest,
    StartAttendanceRequest,
    StopAttendanceRequest,
    TrainModelRequest,
)
from services.attendance_service import attendance_service
from services.enrollment_service import enrollment_service
from services.recognition_service import recognition_service
from utils.camera import generate_mjpeg_stream

router = APIRouter()


@router.post("/register-student")
async def register_student(payload: RegisterStudentRequest):
    try:
        return enrollment_service.register_student(payload)
    except Exception as error:  # pragma: no cover - runtime adapter
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/train-model")
async def train_model(payload: TrainModelRequest):
    try:
        return enrollment_service.train_model(payload)
    except Exception as error:  # pragma: no cover - runtime adapter
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/start-attendance")
async def start_attendance(payload: StartAttendanceRequest):
    try:
        return attendance_service.start_session(payload)
    except Exception as error:  # pragma: no cover - runtime adapter
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/stop-attendance")
async def stop_attendance(payload: StopAttendanceRequest):
    try:
        return attendance_service.stop_session(payload.session_id)
    except Exception as error:  # pragma: no cover - runtime adapter
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/camera-feed")
async def camera_feed():
    return StreamingResponse(
        generate_mjpeg_stream(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.post("/recognize-frame")
async def recognize_frame(payload: RecognizeFrameRequest):
    try:
        return recognition_service.recognize_frame(payload)
    except Exception as error:  # pragma: no cover - runtime adapter
        raise HTTPException(status_code=400, detail=str(error)) from error
