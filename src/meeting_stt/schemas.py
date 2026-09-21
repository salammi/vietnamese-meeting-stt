from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class Classification(BaseModel):
    label: str
    confidence: float = Field(ge=0, le=1)


class TranscriptSegment(BaseModel):
    text: str
    language: str
    started_at_ms: int
    ended_at_ms: int
    confidence: float = Field(ge=0, le=1)
    model: str
    speaker_profile: Classification | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class HealthResponse(BaseModel):
    status: str
    device: str
    models_loaded: bool

