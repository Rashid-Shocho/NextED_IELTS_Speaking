import asyncio
import logging

import httpx
from pathlib import Path
from app.core.config import get_settings

settings = get_settings()
logger = logging.getLogger("asr")


class TranscriptionError(RuntimeError):
    """Raised when transcription fails after all retry attempts. The
    caller (graph/nodes.py transcribe_node) is expected to let this
    propagate -- it becomes a part status='failed' + error_reason, which
    evaluate_session's outer try/except turns into session status='failed'
    with a real error_message, instead of silently scoring the user
    against a fabricated transcript."""


async def _load_audio_bytes(audio_url: str) -> bytes:
    # ---------- 1. Local file ----------
    if not audio_url.startswith(("http://", "https://")):
        local_path = Path(audio_url)
        if not local_path.is_absolute():
            local_path = Path.cwd() / audio_url

        if not local_path.exists():
            raise FileNotFoundError(f"Local audio file not found: {local_path}")

        logger.info("Reading local file: %s", local_path)
        audio_bytes = local_path.read_bytes()
        logger.info("Loaded %d bytes from local file", len(audio_bytes))
        return audio_bytes

    # ---------- 2. Remote URL ----------
    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        logger.info("Downloading audio from: %s", audio_url)
        audio_response = await client.get(audio_url)
        audio_response.raise_for_status()
        audio_bytes = audio_response.content
        logger.info("Downloaded %d bytes", len(audio_bytes))
        return audio_bytes


async def _transcribe_once(audio_bytes: bytes) -> dict:
    headers = {"Authorization": f"Bearer {settings.GROQ_API_KEY}"}

    async with httpx.AsyncClient(timeout=90.0) as client:
        files = {
            "file": ("audio.m4a", audio_bytes, "audio/mp4"),  # m4a works
            "model": (None, settings.GROQ_WHISPER_MODEL),
            "response_format": (None, "verbose_json"),
            "timestamp_granularities[]": (None, "word"),
        }

        response = await client.post(
            "https://api.groq.com/openai/v1/audio/transcriptions",
            headers=headers,
            files=files,
        )
        response.raise_for_status()
        data = response.json()

        words = [
            {"word": w.get("word", ""), "start": w.get("start", 0), "end": w.get("end", 0)}
            for w in (data.get("words", []) or [])
        ]
        transcript = data.get("text", "").strip()
        logger.info("Transcription success - %d chars", len(transcript))
        return {"transcript": transcript, "words": words}


async def transcribe_audio(audio_url: str) -> dict:
    """
    Transcribe audio using Groq Whisper.
    Supports:
      1. Remote URLs (http/https)
      2. Local file paths (e.g. audio_samples/audio1.m4a)

    Retries transient failures (network blips, Groq rate limits/5xx) up to
    TRANSCRIBE_MAX_ATTEMPTS times. If every attempt fails, raises
    TranscriptionError instead of silently returning placeholder text --
    a failed transcription must never be scored as if it were real.
    """
    audio_bytes = await _load_audio_bytes(audio_url)

    last_error: Exception | None = None
    for attempt in range(1, settings.TRANSCRIBE_MAX_ATTEMPTS + 1):
        try:
            return await _transcribe_once(audio_bytes)
        except Exception as e:
            last_error = e
            logger.warning(
                "Transcription attempt %d/%d failed (%s: %s)",
                attempt, settings.TRANSCRIBE_MAX_ATTEMPTS, type(e).__name__, e,
            )
            if attempt < settings.TRANSCRIBE_MAX_ATTEMPTS:
                await asyncio.sleep(settings.TRANSCRIBE_RETRY_BACKOFF_SEC * attempt)

    raise TranscriptionError(
        f"Transcription failed after {settings.TRANSCRIBE_MAX_ATTEMPTS} attempts: "
        f"{type(last_error).__name__}: {last_error}"
    ) from last_error
