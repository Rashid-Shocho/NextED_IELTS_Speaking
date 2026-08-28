# from fastapi import APIRouter, HTTPException, status
# from uuid import uuid4
# from datetime import datetime, timezone
# from fastapi import HTTPException, BackgroundTasks
# from app.models.schemas import ReportResponse, BandScores

# from app.core.database import get_connection
# from app.models.schemas import (
#     CreateSessionRequest,
#     SessionResponse,
#     PartResponse,
#     SessionStatus,
# )
# from app.workers.evaluate import evaluate_session

# router = APIRouter(prefix="/sessions", tags=["Speaking Sessions"])


# @router.post("", response_model=SessionResponse, status_code=status.HTTP_201_CREATED)
# def create_session(payload: CreateSessionRequest):
#     session_id = uuid4()
#     now = datetime.now(timezone.utc)

#     with get_connection() as conn:
#         with conn.cursor() as cur:
#             cur.execute(
#                 """
#                 INSERT INTO speaking_sessions (id, user_id, status, created_at)
#                 VALUES (%s, %s, %s, %s)
#                 """,
#                 (str(session_id), payload.user_id, SessionStatus.pending.value, now),
#             )

#             parts_response = []
#             for p in payload.parts:
#                 part_id = uuid4()
#                 cur.execute(
#                     """
#                     INSERT INTO speaking_parts
#                         (id, session_id, part_number, question_text, audio_url, created_at)
#                     VALUES (%s, %s, %s, %s, %s, %s)
#                     """,
#                     (
#                         str(part_id),
#                         str(session_id),
#                         p.part,
#                         p.question_text,
#                         p.audio_url,
#                         now,
#                     ),
#                 )
#                 parts_response.append(
#                     PartResponse(
#                         id=part_id,
#                         part_number=p.part,
#                         question_text=p.question_text,
#                         audio_url=p.audio_url,
#                     )
#                 )

#             conn.commit()

#     return SessionResponse(
#         id=session_id,
#         user_id=payload.user_id,
#         status=SessionStatus.pending,
#         created_at=now,
#         parts=parts_response,
#     )


# @router.get("/{session_id}", response_model=SessionResponse)
# def get_session(session_id: str):
#     with get_connection() as conn:
#         with conn.cursor() as cur:
#             cur.execute(
#                 """
#                 SELECT id, user_id, status, created_at, completed_at
#                 FROM speaking_sessions
#                 WHERE id = %s
#                 """,
#                 (session_id,),
#             )
#             row = cur.fetchone()
#             if not row:
#                 raise HTTPException(status_code=404, detail="Session not found")

#             cur.execute(
#                 """
#                 SELECT id, part_number, question_text, audio_url,
#                        transcript, fluency_features, pronunciation,
#                        status, error_reason
#                 FROM speaking_parts
#                 WHERE session_id = %s
#                 ORDER BY part_number
#                 """,
#                 (session_id,),
#             )
#             parts_rows = cur.fetchall()

#     parts = [
#         PartResponse(
#             id=r[0],
#             part_number=r[1],
#             question_text=r[2],
#             audio_url=r[3],
#             transcript=r[4],
#             fluency_features=r[5],
#             pronunciation=r[6],
#             status=r[7],
#             error_reason=r[8],
#         )
#         for r in parts_rows
#     ]

#     return SessionResponse(
#         id=row[0],
#         user_id=row[1],
#         status=SessionStatus(row[2]),
#         created_at=row[3],
#         completed_at=row[4],
#         parts=parts,
#     )


# @router.post("/{session_id}/evaluate")
# async def trigger_evaluation(session_id: str, background_tasks: BackgroundTasks):
#     background_tasks.add_task(evaluate_session, session_id)
#     return {"message": "Evaluation started", "session_id": session_id}


# @router.get("/{session_id}/report", response_model=ReportResponse)
# def get_report(session_id: str):
#     with get_connection() as conn:
#         with conn.cursor() as cur:
#             cur.execute(
#                 "SELECT status FROM speaking_sessions WHERE id = %s",
#                 (session_id,),
#             )
#             session_row = cur.fetchone()
#             if not session_row:
#                 raise HTTPException(status_code=404, detail="Session not found")

#             status = session_row[0]

#             cur.execute(
#                 """
#                 SELECT fluency, lexical, grammar, pronunciation, overall,
#                        evidence, created_at
#                 FROM speaking_reports
#                 WHERE session_id = %s
#                 """,
#                 (session_id,),
#             )
#             report_row = cur.fetchone()

#             if not report_row:
#                 raise HTTPException(
#                     status_code=404,
#                     detail=f"Report not ready. Current status: {status}"
#                 )

#     return ReportResponse(
#         session_id=session_id,
#         status=status,
#         scores=BandScores(
#             fluency=float(report_row[0]),
#             lexical=float(report_row[1]),
#             grammar=float(report_row[2]),
#             pronunciation=float(report_row[3]),
#             overall=float(report_row[4]),
#         ),
#         evidence=report_row[5] or {},
#         created_at=report_row[6],
#     )
from fastapi import APIRouter, HTTPException, status
from uuid import uuid4
from datetime import datetime, timezone
from fastapi import HTTPException, BackgroundTasks, UploadFile, File, Form
from app.models.schemas import ReportResponse, BandScores
import shutil
from pathlib import Path

from app.core.database import get_connection
from app.models.schemas import (
    CreateSessionRequest,
    SessionResponse,
    PartResponse,
    SessionStatus,
)
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
                cur.execute(
                    """
                    INSERT INTO speaking_parts
                        (id, session_id, part_number, question_text, audio_url, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(part_id),
                        str(session_id),
                        p.part,
                        p.question_text,
                        p.audio_url,
                        now,
                    ),
                )
                parts_response.append(
                    PartResponse(
                        id=part_id,
                        part_number=p.part,
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
async def submit_session(
    background_tasks: BackgroundTasks,
    user_id: str = Form(None),
    part1_question: str = Form(...),
    part2_question: str = Form(...),
    part3_question: str = Form(...),
    part1_audio: UploadFile = File(...),
    part2_audio: UploadFile = File(...),
    part3_audio: UploadFile = File(...),
):
    session_id = uuid4()
    now = datetime.now(timezone.utc)
    session_dir = UPLOAD_DIR / str(session_id)
    session_dir.mkdir(parents=True, exist_ok=True)

    parts_input = [
        (1, part1_question, part1_audio),
        (2, part2_question, part2_audio),
        (3, part3_question, part3_audio),
    ]

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
            for part_number, question_text, audio_file in parts_input:
                suffix = Path(audio_file.filename or "audio.webm").suffix or ".webm"
                saved_path = session_dir / f"part{part_number}{suffix}"

                with saved_path.open("wb") as f:
                    shutil.copyfileobj(audio_file.file, f)

                part_id = uuid4()
                cur.execute(
                    """
                    INSERT INTO speaking_parts
                        (id, session_id, part_number, question_text, audio_url, created_at)
                    VALUES (%s, %s, %s, %s, %s, %s)
                    """,
                    (
                        str(part_id),
                        str(session_id),
                        part_number,
                        question_text,
                        str(saved_path),
                        now,
                    ),
                )
                parts_response.append(
                    PartResponse(
                        id=part_id,
                        part_number=part_number,
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
                SELECT id, part_number, question_text, audio_url,
                       transcript, fluency_features, pronunciation,
                       status, error_reason
                FROM speaking_parts
                WHERE session_id = %s
                ORDER BY part_number
                """,
                (session_id,),
            )
            parts_rows = cur.fetchall()

    parts = [
        PartResponse(
            id=r[0],
            part_number=r[1],
            question_text=r[2],
            audio_url=r[3],
            transcript=r[4],
            fluency_features=r[5],
            pronunciation=r[6],
            status=r[7],
            error_reason=r[8],
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