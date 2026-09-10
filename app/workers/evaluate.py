import asyncio
from datetime import datetime, timezone
from pathlib import Path

from psycopg.types.json import Jsonb

from app.core.database import get_connection
from app.graph.workflow import part_workflow
from app.services.llm_scorer import score_session


async def run_part(part_id: str, session_id: str, part_number: int, segment_id: str, label: str,
                    question_text: str, audio_url: str, cloud_audio_url: str | None) -> dict:
    """Run one segment through the LangGraph part workflow (VAD gate ->
    transcribe+pronunciation -> finalize), see app/graph/workflow.py."""
    initial_state = {
        "part_id": part_id,
        "session_id": session_id,
        "part_number": part_number,
        "question_text": question_text,
        "audio_url": audio_url,
    }
    final_state = await part_workflow.ainvoke(initial_state)

    return {
        "part_id": part_id,
        "part_number": part_number,
        "segment_id": segment_id,
        "label": label,
        "question_text": question_text,
        "audio_url": audio_url,
        "cloud_audio_url": cloud_audio_url,
        "status": final_state.get("status"),
        "error_reason": final_state.get("error_reason"),
        "transcript": final_state.get("transcript", ""),
        "fluency_features": final_state.get("fluency_features"),
        "pronunciation_result": final_state.get("pronunciation_result"),
    }


async def evaluate_session(session_id: str):
    """
    1. Mark session as processing
    2. Run every part through the per-part graph concurrently
       (VAD gate -> transcribe+pronunciation -> finalize)
    3. If ANY part came back no_speech_detected -> stop here, mark the
       session needs_rerecording, SKIP the [LLM] call entirely
    4. Otherwise, single session-level Groq gpt-oss-20b call (Pattern A)
       + report, then mark session completed.
       No embedding step -- the architecture removed vector search entirely.
    """
    print(f"[Worker] Starting evaluation for session {session_id}")

    try:
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "UPDATE speaking_sessions SET status = 'processing' WHERE id = %s",
                    (session_id,),
                )
                conn.commit()

                cur.execute(
                    "SELECT user_id FROM speaking_sessions WHERE id = %s",
                    (session_id,),
                )
                session_row = cur.fetchone()
                user_id = session_row[0] if session_row else None

                cur.execute(
                    """
                    SELECT id, part_number, segment_id, label, question_text, audio_url, cloud_audio_url
                    FROM speaking_parts
                    WHERE session_id = %s
                    ORDER BY part_number, segment_id
                    """,
                    (session_id,),
                )
                parts = cur.fetchall()

        if not parts:
            raise ValueError(f"No parts found for session {session_id}")

        tasks = [
            run_part(str(p[0]), session_id, p[1], p[2] or f"part{p[1]}", p[3] or f"Part {p[1]}", p[4], p[5], p[6])
            for p in parts
        ]
        results = await asyncio.gather(*tasks)

        # --- has_speech gate at the session level ---
        no_speech_parts = [r for r in results if r["status"] == "no_speech_detected"]

        if no_speech_parts:
            labels = ", ".join(r["label"] for r in no_speech_parts)
            message = f"{labels} had no detectable speech. Please re-record and resubmit."

            with get_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        UPDATE speaking_sessions
                        SET status = 'needs_rerecording', error_message = %s
                        WHERE id = %s
                        """,
                        (message, session_id),
                    )
                    conn.commit()

            print(f"[Worker] Session {session_id} needs re-recording: {message}")
            return  # SKIP [LLM] call entirely

        # --- all segments have speech -> single session-level [LLM] call ---
        # Group segments back under their parent part (1/2/3) so the LLM
        # sees e.g. Part 1's intro + 3 topic cards as one coherent Part 1,
        # while each segment keeps its own question/transcript/evidence
        # pairing rather than being merged into one blob.
        parts_grouped: dict[int, list[dict]] = {}
        for r in results:
            parts_grouped.setdefault(r["part_number"], []).append(r)

        scores = await score_session(parts_grouped)

        # speaking_reports only has dedicated columns for the 5 band scores --
        # fold generalSummary/keyImprovements into the evidence JSON blob so
        # they persist and round-trip through GET /sessions/{id}/report.
        evidence_to_store = {
            **scores.get("evidence", {}),
            "generalSummary": scores.get("generalSummary", ""),
            "keyImprovements": scores.get("keyImprovements", []),
            "partsRecorded": sorted(parts_grouped.keys()),
            "segmentsRecorded": [r["segment_id"] for r in results],
        }

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO speaking_reports (
                        session_id,
                        fluency, lexical, grammar, pronunciation, overall,
                        evidence, created_at
                    )
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    ON CONFLICT (session_id) DO UPDATE SET
                        fluency = EXCLUDED.fluency,
                        lexical = EXCLUDED.lexical,
                        grammar = EXCLUDED.grammar,
                        pronunciation = EXCLUDED.pronunciation,
                        overall = EXCLUDED.overall,
                        evidence = EXCLUDED.evidence,
                        created_at = EXCLUDED.created_at
                    """,
                    (
                        session_id,
                        scores["fluency"],
                        scores["lexical"],
                        scores["grammar"],
                        scores["pronunciation"],
                        scores["overall"],
                        Jsonb(evidence_to_store),
                        datetime.now(timezone.utc),
                    ),
                )

                cur.execute(
                    """
                    UPDATE speaking_sessions
                    SET status = 'completed', completed_at = %s
                    WHERE id = %s
                    """,
                    (datetime.now(timezone.utc), session_id),
                )

                # One pending row per part for the future training dataset.
                # expert_* fields are left NULL for a human examiner to fill
                # in later; consent_given defaults to false (schema default)
                # -- flipping that to true is a product/consent-flow
                # decision, not something this worker should do silently.
                # audio_url here prefers the R2 URL; if R2 wasn't configured
                # or the upload failed, falls back to the local path so the
                # row still exists and can be backfilled later.
                for r in results:
                    if r["status"] != "pronunciation_done":
                        continue  # defensive -- the no_speech gate above already covers this
                    if not user_id:
                        # speaking_training_samples.user_id is NOT NULL --
                        # skip rather than fail this whole commit (which
                        # would also roll back the report/completed status
                        # that already succeeded above) for anonymous
                        # sessions with no logged-in user.
                        continue
                    audio_url_for_dataset = r.get("cloud_audio_url") or r["audio_url"]
                    audio_format = Path(r["audio_url"]).suffix.lstrip(".").lower() or None
                    fluency_features = r.get("fluency_features") or {}
                    cur.execute(
                        """
                        INSERT INTO speaking_training_samples (
                            user_id, session_id, part_id, part_number,
                            question_text, audio_url, audio_duration_seconds, audio_format,
                            transcript_text, transcript_source,
                            annotation_status
                        )
                        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            user_id,
                            session_id,
                            r["part_id"],
                            r["part_number"],
                            r["question_text"],
                            audio_url_for_dataset,
                            fluency_features.get("total_duration_sec"),
                            audio_format,
                            r["transcript"],
                            "whisper",
                            "pending",
                        ),
                    )

                conn.commit()

        print(f"[Worker] Session {session_id} completed. Overall band: {scores['overall']}")

    except Exception as e:
        print(f"[Worker] Error evaluating session {session_id}: {e}")
        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE speaking_sessions
                    SET status = 'failed', error_message = %s
                    WHERE id = %s
                    """,
                    (str(e), session_id),
                )
                conn.commit()
        raise