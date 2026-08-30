from fastapi import APIRouter, HTTPException, status
from uuid import uuid4
from datetime import datetime, timezone
from fastapi import HTTPException, BackgroundTasks, Request
from app.models.schemas import ReportResponse, BandScores
import asyncio
import json
from pathlib import Path

from app.core.database import get_connection
from app.models.schemas import (
    CreateSessionRequest,
    SessionResponse,
    PartResponse,
    SessionStatus,
)
from app.services.storage import upload_audio_to_r2
from app.workers.evaluate import evaluate_session

router = APIRouter(prefix="/sessions", tags=["Speaking Sessions"])

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
def create_session(payload: CreateSessionRequest):
    session_id = uuid4()
    now = datetime.now(timezone.utc)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO speaking_sessions (id, user_id, status, created_at)
                VALUES (%s, %s, %s, %s)
                """,
                (str(session_id), payload.user_id, SessionStatus.pending.value, now),
            )

            parts_response = []
            for p in payload.parts:
                part_id = uuid4()
                seg_id = p.segment_id or f"part{p.part}"
                label = p.label or f"Part {p.part}"
                cur.execute(
                    """
                    INSERT INTO speaking_parts
                        (id, session_id, part_number, segment_id, label, question_text, audio_url, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(part_id),
                        str(session_id),
                        p.part,
                        seg_id,
                        label,
                        p.question_text,
                        p.audio_url,
                        now,
                    ),
                )
                parts_response.append(
                    PartResponse(
                        id=part_id,
                        part_number=p.part,
                        segment_id=seg_id,
                        label=label,
                        question_text=p.question_text,
                        audio_url=p.audio_url,
                    )
                )

            conn.commit()

    return SessionResponse(
        id=session_id,
        user_id=payload.user_id,
        status=SessionStatus.pending,
        created_at=now,
        parts=parts_response,
    )


@router.post("/submit", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
async def submit_session(background_tasks: BackgroundTasks, request: Request):
    """
    Accepts a variable number of segments instead of a fixed 3 parts, since
    Part 1 and Part 3 are now each split into several separately-recorded
    sub-questions (e.g. Part 1 = intro + 3 topic cards).

    multipart/form-data body:
      - user_id: optional string
      - segments_meta: JSON string, a list of
            {"id": "p1_intro", "part_number": 1, "label": "...", "question_text": "..."}
      - audio_<id>: one file field per segment in segments_meta, e.g. audio_p1_intro

    Every segment listed in segments_meta must have a matching audio_<id>
    file; at least one segment is required.
    """
    form = await request.form()

    user_id = form.get("user_id") or None
    raw_meta = form.get("segments_meta")
    if not raw_meta:
        raise HTTPException(status_code=400, detail="segments_meta is required.")

    try:
        meta_list = json.loads(raw_meta)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=400, detail="segments_meta must be valid JSON.")

    if not isinstance(meta_list, list) or not meta_list:
        raise HTTPException(status_code=400, detail="segments_meta must be a non-empty list.")

    segments_input = []
    for entry in meta_list:
        seg_id = entry.get("id")
        part_number = entry.get("part_number")
        label = entry.get("label") or seg_id
        question_text = entry.get("question_text")

        if not seg_id or part_number not in (1, 2, 3) or not question_text:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid segment metadata entry: {entry!r} "
                       f"(needs id, part_number in 1-3, question_text).",
            )

        audio_file = form.get(f"audio_{seg_id}")
        if audio_file is None or not hasattr(audio_file, "filename"):
            raise HTTPException(status_code=400, detail=f"Missing audio file for segment '{seg_id}'.")

        segments_input.append((seg_id, part_number, label, question_text, audio_file))

    session_id = uuid4()
    now = datetime.now(timezone.utc)
    session_dir = UPLOAD_DIR / str(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO speaking_sessions (id, user_id, status, created_at)
                VALUES (%s, %s, %s, %s)
                """,
                (str(session_id), user_id, SessionStatus.pending.value, now),
            )

            parts_response = []
            for seg_id, part_number, label, question_text, audio_file in segments_input:
                suffix = Path(audio_file.filename or "audio.webm").suffix or ".webm"
                saved_path = session_dir / f"{seg_id}{suffix}"

                contents = await audio_file.read()
                with saved_path.open("wb") as f:
                    f.write(contents)

                # Best-effort archival to R2 for the future training dataset.
                # Never blocks or fails the submission -- returns None (and
                # logs why) if R2 isn't configured or the upload errors.
                # Local disk stays the source of truth for the live pipeline
                # either way (VAD/ASR/pronunciation keep reading saved_path).
                r2_key = f"speaking/{session_id}/{seg_id}{suffix}"
                cloud_audio_url = await asyncio.to_thread(upload_audio_to_r2, saved_path, r2_key)

                part_id = uuid4()
                cur.execute(
                    """
                    INSERT INTO speaking_parts
                        (id, session_id, part_number, segment_id, label, question_text,
                         audio_url, cloud_audio_url, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(part_id),
                        str(session_id),
                        part_number,
                        seg_id,
                        label,
                        question_text,
                        str(saved_path),
                        cloud_audio_url,
                        now,
                    ),
                )
                parts_response.append(
                    PartResponse(
                        id=part_id,
                        part_number=part_number,
                        segment_id=seg_id,
                        label=label,
                        question_text=question_text,
                        audio_url=str(saved_path),
                    )
                )

            conn.commit()

    background_tasks.add_task(evaluate_session, str(session_id))

    return SessionResponse(
        id=session_id,
        user_id=user_id,
        status=SessionStatus.pending,
        created_at=now,
        parts=parts_response,
    )


@router.get("/{session_id}", response_model=SessionResponse)
def get_session(session_id: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT id, user_id, status, created_at, completed_at
                FROM speaking_sessions
                WHERE id = %s
                """,
                (session_id,),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="Session not found")

            cur.execute(
                """
                SELECT id, part_number, segment_id, label, question_text, audio_url,
                       transcript, fluency_features, pronunciation,
                       status, error_reason
                FROM speaking_parts
                WHERE session_id = %s
                ORDER BY part_number, segment_id
                """,
                (session_id,),
            )
            parts_rows = cur.fetchall()

    parts = [
        PartResponse(
            id=r[0],
            part_number=r[1],
            segment_id=r[2] or f"part{r[1]}",
            label=r[3] or f"Part {r[1]}",
            question_text=r[4],
            audio_url=r[5],
            transcript=r[6],
            fluency_features=r[7],
            pronunciation=r[8],
            status=r[9],
            error_reason=r[10],
        )
        for r in parts_rows
    ]

    return SessionResponse(
        id=row[0],
        user_id=row[1],
        status=SessionStatus(row[2]),
        created_at=row[3],
        completed_at=row[4],
        parts=parts,
    )


@router.post("/{session_id}/evaluate")
async def trigger_evaluation(session_id: str, background_tasks: BackgroundTasks):
    background_tasks.add_task(evaluate_session, session_id)
    return {"message": "Evaluation started", "session_id": session_id}


@router.get("/{session_id}/report", response_model=ReportResponse)
def get_report(session_id: str):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT status FROM speaking_sessions WHERE id = %s",
                (session_id,),
            )
            session_row = cur.fetchone()
            if not session_row:
                raise HTTPException(status_code=404, detail="Session not found")

            status = session_row[0]

            cur.execute(
                """
                SELECT fluency, lexical, grammar, pronunciation, overall,
                       evidence, created_at
                FROM speaking_reports
                WHERE session_id = %s
                """,
                (session_id,),
            )
            report_row = cur.fetchone()

            if not report_row:
                raise HTTPException(
                    status_code=404,
                    detail=f"Report not ready. Current status: {status}"
                )

    return ReportResponse(
        session_id=session_id,
        status=status,
        scores=BandScores(
            fluency=float(report_row[0]),
            lexical=float(report_row[1]),
            grammar=float(report_row[2]),
            pronunciation=float(report_row[3]),
            overall=float(report_row[4]),
        ),
        evidence=report_row[5] or {},
        created_at=report_row[6],
    )