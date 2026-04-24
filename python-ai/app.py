import asyncio
from collections import deque
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import json
import logging
import os
import time
import uuid

from routes.attendance_routes import router as attendance_router
from services.attendance_service import attendance_service
from services.recognition_service import recognition_service
from utils.config import settings
from utils.storage import ensure_storage


logging.basicConfig(
    level=os.getenv("AI_LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
LOGGER = logging.getLogger("facetrace.ai")
APP_INSTANCE_ID = str(uuid.uuid4())
BOOT_TIME = time.time()
RUNTIME_EVENTS = deque(maxlen=120)

app = FastAPI(
    title="FaceTrace AI Service",
    description="FastAPI microservice for MediaPipe-based face enrollment, recognition, and attendance orchestration.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _record_runtime_event(event_type: str, payload: dict) -> None:
    RUNTIME_EVENTS.appendleft(
        {
            "event_id": str(uuid.uuid4()),
            "type": event_type,
            "timestamp": time.time(),
            "payload": payload,
        }
    )


def _build_health_payload() -> dict:
    dataset_root = settings.datasets_dir
    embedding_root = settings.embeddings_dir
    student_count = sum(1 for path in dataset_root.iterdir() if path.is_dir()) if dataset_root.exists() else 0
    uptime_seconds = round(time.time() - BOOT_TIME, 2)

    return {
        "status": "ok",
        "message": "Python AI service is healthy.",
        "engine": settings.default_engine,
        "connected": True,
        "instance_id": APP_INSTANCE_ID,
        "pid": os.getpid(),
        "uptime_seconds": uptime_seconds,
        "storage": {
            "datasets_dir": dataset_root.as_posix(),
            "embeddings_dir": embedding_root.as_posix(),
            "student_profiles": student_count,
        },
        "session": attendance_service.get_state(),
        "recent_events": recognition_service.get_recent_events(limit=10),
        "runtime_events": list(RUNTIME_EVENTS)[:10],
    }


@app.on_event("startup")
async def on_startup():
    ensure_storage()
    _record_runtime_event(
        "startup",
        {
            "engine": settings.default_engine,
            "camera_index": settings.camera_index,
        },
    )
    LOGGER.info("FaceTrace AI service started with engine '%s'.", settings.default_engine)


@app.get("/health")
async def health_check(background_tasks: BackgroundTasks):
    background_tasks.add_task(
        _record_runtime_event,
        "health_check",
        {"instance_id": APP_INSTANCE_ID},
    )
    return JSONResponse(content=json.loads(json.dumps(_build_health_payload())))


@app.get("/events/{limit}")
async def recent_events(limit: int):
    if limit < 1 or limit > 100:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 100.")

    return JSONResponse(
        content={
            "status": "ok",
            "events": recognition_service.get_recent_events(limit=limit),
        }
    )


@app.websocket("/ws/health")
async def health_socket(websocket: WebSocket):
    await websocket.accept()
    _record_runtime_event("websocket_connected", {"client": str(websocket.client)})

    try:
        while True:
            await websocket.send_text(json.dumps(_build_health_payload()))
            await asyncio.sleep(2)
    except Exception as error:  # pragma: no cover - runtime connection lifecycle
        LOGGER.info("Health websocket closed: %s", error)
    finally:
        _record_runtime_event("websocket_disconnected", {"client": str(websocket.client)})


app.include_router(attendance_router)
