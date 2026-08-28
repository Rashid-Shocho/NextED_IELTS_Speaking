import asyncio
from datetime import datetime, timezone

from psycopg.types.json import Jsonb

from app.core.database import get_connection
from app.graph.workflow import part_workflow
from app.services.llm_scorer import score_session


async def run_part(part_id: str, session_id: str, part_number: int,
                    question_text: str, audio_url: str) -> dict:
    """Run one part through the LangGraph part workflow (VAD gate ->
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
        "question_text": question_text,
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
                    """
                    SELECT id, part_number, question_text, audio_url
                    FROM speaking_parts
                    WHERE session_id = %s
                    ORDER BY part_number
                    """,
                    (session_id,),
                )
                parts = cur.fetchall()

        if not parts:
            raise ValueError(f"No parts found for session {session_id}")

        tasks = [
            run_part(str(p[0]), session_id, p[1], p[2], p[3])
            for p in parts
        ]
        results = await asyncio.gather(*tasks)

        # --- has_speech gate at the session level ---
        no_speech_parts = [r for r in results if r["status"] == "no_speech_detected"]

        if no_speech_parts:
            part_numbers = ", ".join(str(r["part_number"]) for r in no_speech_parts)
            message = f"Part(s) {part_numbers} had no detectable speech. Please re-record and resubmit."

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

        # --- all parts have speech -> single session-level [LLM] call ---
        questions = [r["question_text"] for r in results]
        transcripts = [r["transcript"] for r in results]
        fluency_list = [r["fluency_features"] for r in results]
        pron_list = [r["pronunciation_result"] for r in results]

        scores = await score_session(
            questions=questions,
            transcripts=transcripts,
            fluency_features=fluency_list,
            pronunciation_evidence=pron_list,
        )

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
                        Jsonb(scores["evidence"]),
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