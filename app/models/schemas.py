from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, List
from uuid import UUID
from datetime import datetime
from enum import Enum


class SessionStatus(str, Enum):
    pending = "pending"
    processing = "processing"
    completed = "completed"
    failed = "failed"
    needs_rerecording = "needs_rerecording"   # one or more parts had no detectable speech


class PartStatus(str, Enum):
    pending = "pending"
    transcribing = "transcribing"
    no_speech_detected = "no_speech_detected"
    pronunciation_done = "pronunciation_done"
    failed = "failed"


class PartInput(BaseModel):
    part: int = Field(..., ge=1, le=3)
    question_text: str
    audio_url: str
    segment_id: Optional[str] = None
    label: Optional[str] = None


class CreateSessionRequest(BaseModel):
    user_id: Optional[str] = None
    parts: List[PartInput] = Field(..., min_length=1, max_length=12)


class PartResponse(BaseModel):
    id: UUID
    part_number: int
    segment_id: str
    label: str
    question_text: str
    audio_url: str
    transcript: Optional[str] = None
    fluency_features: Optional[dict] = None
    pronunciation: Optional[dict] = None
    status: PartStatus = PartStatus.pending
    error_reason: Optional[str] = None


class SessionResponse(BaseModel):
    id: UUID
    user_id: Optional[str]
    status: SessionStatus
    created_at: datetime
    completed_at: Optional[datetime] = None
    parts: List[PartResponse] = []


class BandScores(BaseModel):
    fluency: float
    lexical: float
    grammar: float
    pronunciation: float
    overall: float


class ReportResponse(BaseModel):
    session_id: UUID
    scores: BandScores
    status: str
    evidence: dict
    created_at: datetime