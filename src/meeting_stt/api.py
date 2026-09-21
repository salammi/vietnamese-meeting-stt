from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .config import Settings, get_settings
from .schemas import HealthResponse
from .service import TranscriptionService
from .vad import UtteranceDetector

logger = logging.getLogger(__name__)
STATIC_DIR = Path(__file__).parent / "static"


def create_app(settings: Settings | None = None, load_models: bool = True) -> FastAPI:
    settings = settings or get_settings()
    service = TranscriptionService(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = service
        if load_models:
            await asyncio.to_thread(service.load)
        yield

    app = FastAPI(
        title="Vietnamese Meeting STT",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

    @app.get("/", include_in_schema=False)
    async def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/api/health", response_model=HealthResponse)
    async def health() -> HealthResponse:
        return HealthResponse(
            status="ok" if service.loaded else "starting",
            device=settings.resolved_device,
            models_loaded=service.loaded,
        )

    @app.websocket("/ws/transcribe")
    async def transcribe_socket(websocket: WebSocket) -> None:
        await websocket.accept()
        detector = UtteranceDetector(
            sample_rate=settings.sample_rate,
            aggressiveness=settings.vad_aggressiveness,
            min_utterance_ms=settings.min_utterance_ms,
            end_silence_ms=settings.end_silence_ms,
            max_utterance_seconds=settings.max_utterance_seconds,
            pre_roll_ms=settings.pre_roll_ms,
        )
        await websocket.send_json(
            {"type": "ready", "sampleRate": settings.sample_rate, "frameMs": 20}
        )
        try:
            while True:
                message = await websocket.receive()
                if message.get("type") == "websocket.disconnect":
                    break
                if message.get("bytes") is not None:
                    for utterance in detector.feed(message["bytes"]):
                        await websocket.send_json({"type": "processing"})
                        segment = await service.transcribe(utterance)
                        await websocket.send_text(
                            json.dumps(
                                {"type": "transcript", "segment": segment.model_dump(mode="json")},
                                ensure_ascii=False,
                            )
                        )
                elif message.get("text") == "flush":
                    utterance = detector.flush()
                    if utterance:
                        segment = await service.transcribe(utterance)
                        await websocket.send_text(
                            json.dumps(
                                {"type": "transcript", "segment": segment.model_dump(mode="json")},
                                ensure_ascii=False,
                            )
                        )
        except WebSocketDisconnect:
            logger.info("Meeting client disconnected")

    return app


app = create_app()
