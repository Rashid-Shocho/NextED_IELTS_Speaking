"""
Per-part evaluation nodes:

    Prepare audio
        -> VAD pass (Silero VAD): speech/silence segments
        -> has_speech check
            no speech  -> status = no_speech_detected
                          SKIP transcription + pronunciation + [LLM]
                          surfaced to caller: re-record this part
            has speech -> transcribe (Whisper via Groq)
                       -> pronunciation (wav2vec2 + forced_align),
                          now SEQUENTIAL after transcribe, not parallel --
                          the RunPod handler needs reference_text (the
                          transcript) to forced-align against, so it can't
                          run until transcribe has produced one.
                       -> finalize: combine VAD segments + word count
                          into fluency features, store per-phoneme
                          pronunciation evidence, store part evidence
                       -> status = pronunciation_done
                          (ready for session-level [LLM] scoring
                          once all valid parts are done)

No embedding model, no vector search anywhere in this flow (Appendix A).
Session-level [LLM] scoring (Groq gpt-oss-20b, one call per session) is
triggered by app/workers/evaluate.py once all parts in a session have
finished this graph.
"""

from typing import Optional, TypedDict

from psycopg.types.json import Jsonb

from app.core.database import get_connection
from app.services.vad import detect_speech_segments
from app.services.asr import transcribe_audio
from app.services.pronunciation import analyze_pronunciation


class PartState(TypedDict, total=False):
    part_id: str
    session_id: str
    part_number: int
    question_text: str
    audio_url: str

    has_speech: bool
    segments: list
    total_duration_sec: float

    transcript: str
    words: list
    pronunciation_result: dict

    fluency_features: dict
    status: str            # pending | transcribing | no_speech_detected | pronunciation_done | failed
    error_reason: Optional[str]


# ---------------------------------------------------------------------------
# Node: vad  (Silero VAD pass)
# ---------------------------------------------------------------------------
async def vad_node(state: PartState) -> dict:
    vad_result = await detect_speech_segments(state["audio_url"])
    return {
        "has_speech": vad_result["has_speech"],
        "segments": vad_result["segments"],
        "total_duration_sec": vad_result["total_duration_sec"],
        "status": "transcribing",
    }


# ---------------------------------------------------------------------------
# Router: has_speech check (VAD-based, not Whisper-word-based)
# ---------------------------------------------------------------------------
def route_after_vad(state: PartState) -> str:
    if not state.get("has_speech"):
        return "mark_no_speech"
    return "transcribe"


# ---------------------------------------------------------------------------
# Node: mark_no_speech
# ---------------------------------------------------------------------------
async def mark_no_speech_node(state: PartState) -> dict:
    """SKIP transcription + pronunciation + [LLM]. Persist immediately."""
    error_reason = "No speech detected in audio - please re-record this part."

    _persist_part(
        part_id=state["part_id"],
        status="no_speech_detected",
        transcript="",
        fluency_features=None,
        pronunciation=None,
        error_reason=error_reason,
    )

    return {
        "status": "no_speech_detected",
        "error_reason": error_reason,
    }


# ---------------------------------------------------------------------------
# Node: transcribe  (Whisper, via Groq)
# ---------------------------------------------------------------------------
async def transcribe_node(state: PartState) -> dict:
    asr_result = await transcribe_audio(state["audio_url"])
    return {
        "transcript": (asr_result.get("transcript") or "").strip(),
        "words": asr_result.get("words") or [],
    }


# ---------------------------------------------------------------------------
# Node: pronunciation  (wav2vec2-lv-60-espeak-cv-ft + forced_align via RunPod)
# ---------------------------------------------------------------------------
async def pronunciation_node(state: PartState) -> dict:
    """
    Runs AFTER transcribe (see workflow.py) -- the RunPod handler needs
    reference_text (the transcript) to forced-align against.
    """
    pron_result = await analyze_pronunciation(
        audio_url=state["audio_url"],
        reference_text=state.get("transcript", ""),
    )
    return {"pronunciation_result": pron_result}


# ---------------------------------------------------------------------------
# Node: finalize (runs after pronunciation)
# ---------------------------------------------------------------------------
async def finalize_node(state: PartState) -> dict:
    """
    Combine VAD segments (already in state, from vad_node) with the word
    count from the transcript into final fluency features, then store
    everything: part row + pronunciation evidence.
    """
    words = state.get("words", [])
    fluency = compute_fluency_features(
        segments=state.get("segments", []),
        total_duration_sec=state.get("total_duration_sec", 0),
        word_count=len(words),
    )

    pronunciation_result = state.get("pronunciation_result") or {}

    _persist_part(
        part_id=state["part_id"],
        status="pronunciation_done",
        transcript=state.get("transcript", ""),
        fluency_features=fluency,
        pronunciation=pronunciation_result,
        error_reason=None,
    )

    # NOTE: pronunciation_parser.analyze_gop_result() returns
    # phoneme-level evidence (severe_flags, worst_phoneme), not
    # per-WORD scores -- the real GOP model doesn't group phonemes back
    # into words. pronunciation_words stays empty for now; revisit if
    # word-level grouping becomes available later (would need the
    # forced-alignment output's word boundaries, not just phoneme IDs).

    return {
        "fluency_features": fluency,
        "status": "pronunciation_done",
    }


def compute_fluency_features(segments: list, total_duration_sec: float, word_count: int) -> dict:
    """
    Deterministic fluency features (no model). speech_rate/articulation_rate
    need the transcript's word count, so this only finalizes after both the
    VAD pass and the transcript are available -- see finalize_node.
    """
    if not segments or total_duration_sec <= 0:
        return {
            "speech_rate_wpm": 0,
            "articulation_rate_wpm": 0,
            "phonation_time_ratio": 0,
            "pause_count": 0,
            "pause_total_sec": 0,
            "total_duration_sec": total_duration_sec,
            "word_count": word_count,
        }

    phonation_time = sum(s["end"] - s["start"] for s in segments)
    phonation_time_ratio = phonation_time / total_duration_sec if total_duration_sec else 0

    pause_count = 0
    pause_total = 0.0
    for i in range(1, len(segments)):
        gap = segments[i]["start"] - segments[i - 1]["end"]
        if gap > 0.3:
            pause_count += 1
            pause_total += gap

    speech_rate_wpm = (word_count / total_duration_sec) * 60 if total_duration_sec else 0
    articulation_rate_wpm = (word_count / phonation_time) * 60 if phonation_time > 0 else 0

    return {
        "speech_rate_wpm": round(speech_rate_wpm, 1),
        "articulation_rate_wpm": round(articulation_rate_wpm, 1),
        "phonation_time_ratio": round(phonation_time_ratio, 3),
        "pause_count": pause_count,
        "pause_total_sec": round(pause_total, 2),
        "total_duration_sec": round(total_duration_sec, 2),
        "word_count": word_count,
    }


# ---------------------------------------------------------------------------
# DB helper
# ---------------------------------------------------------------------------
def _persist_part(
    part_id: str,
    status: str,
    transcript: str,
    fluency_features,
    pronunciation,
    error_reason,
) -> None:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE speaking_parts
                SET transcript = %s,
                    fluency_features = %s,
                    pronunciation = %s,
                    status = %s,
                    error_reason = %s
                WHERE id = %s
                """,
                (
                    transcript,
                    Jsonb(fluency_features) if fluency_features is not None else None,
                    Jsonb(pronunciation) if pronunciation is not None else None,
                    status,
                    error_reason,
                    part_id,
                ),
            )
            conn.commit()