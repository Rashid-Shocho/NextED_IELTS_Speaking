"""
Pronunciation scoring via the RunPod endpoint. All GOP text-processing
logic (parsing, dedup, severity flagging) lives in
pronunciation_parser.py -- this file only handles the network call and
hands whatever comes back to analyze_gop_result().

Two things the RunPod handler requires that this must supply:
1. reference_text -- the handler forced-aligns the phoneme recognizer's
   output against this text, so it can't score without it. We use the
   Whisper transcript as the reference text (see nodes.py: pronunciation
   now runs AFTER transcribe, not in parallel with it).
2. audio as base64, not a URL -- local file paths (audio_samples/*.m4a)
   aren't reachable from RunPod's cloud workers. We read the file and
   base64-encode it instead. (Swap this for a real audio_url once files
   are hosted somewhere reachable, e.g. S3/R2 -- see the TODO below.)
"""

import asyncio
import base64
import re
from pathlib import Path

import httpx

from app.core.config import get_settings
from app.services.pronunciation_parser import analyze_gop_result

settings = get_settings()

ALLOWED_AUDIO_FORMATS = {"wav", "mp3", "m4a", "ogg", "flac"}


def _status_url(run_url: str, job_id: str) -> str:
    return re.sub(r"/run/?$", f"/status/{job_id}", run_url)


_DEFAULT_RESULT = {
    "utterance_avg": None,
    "total_phonemes": 0,
    "distribution": {"excellent": 0, "good": 0, "moderate": 0, "severe": 0},
    "severe_flags": [],
    "worst_phoneme": None,
}


async def _load_audio_b64_and_format(audio_url: str) -> tuple[str, str]:
    """
    Reads audio (currently only local paths are handled) and returns
    (base64_string, format_extension). audio_url is still the parameter
    name for now since that's what the rest of the pipeline calls it --
    despite the name, it's treated as a local file path here.

    TODO: once audio is hosted somewhere RunPod's workers can reach
    (S3/Cloudflare R2/presigned URL), send {"audio_url": ...} directly
    instead of reading+base64-encoding, and skip this function entirely.
    """
    local_path = Path(audio_url)
    if not local_path.is_absolute():
        local_path = Path.cwd() / audio_url
    if not local_path.exists():
        raise FileNotFoundError(f"Local audio file not found: {local_path}")

    audio_format = local_path.suffix.lstrip(".").lower() or "wav"
    if audio_format not in ALLOWED_AUDIO_FORMATS:
        raise ValueError(f"audio_format '{audio_format}' not in {ALLOWED_AUDIO_FORMATS}")

    audio_bytes = local_path.read_bytes()
    audio_b64 = base64.b64encode(audio_bytes).decode("ascii")
    return audio_b64, audio_format


async def analyze_pronunciation(audio_url: str, reference_text: str) -> dict:
    """
    audio_url: local file path (see _load_audio_b64_and_format's TODO to
               switch this to a real reachable URL later).
    reference_text: the transcript to forced-align against (from
                     transcribe_node -- pronunciation now runs after
                     transcription, not in parallel with it).
    """
    if not reference_text or not reference_text.strip():
        print("[Pronunciation] No reference_text (empty transcript) -- skipping RunPod call")
        return _DEFAULT_RESULT

    headers = {
        "Authorization": f"Bearer {settings.RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    try:
        audio_b64, audio_format = await _load_audio_b64_and_format(audio_url)

        async with httpx.AsyncClient(timeout=60.0) as client:
            submit_resp = await client.post(
                settings.RUNPOD_PRONUNCIATION_URL,
                headers=headers,
                json={
                    "input": {
                        "audio_base64": audio_b64,
                        "audio_format": audio_format,
                        "reference_text": reference_text,
                    }
                },
            )
            submit_resp.raise_for_status()
            submit_data = submit_resp.json()

            job_id = submit_data.get("id")
            if not job_id:
                raise ValueError(f"RunPod response had no job id: {submit_data}")

            status = submit_data.get("status")
            output = submit_data.get("output")

            if status != "COMPLETED":
                status_url = _status_url(settings.RUNPOD_PRONUNCIATION_URL, job_id)
                elapsed = 0.0

                while elapsed < settings.RUNPOD_POLL_TIMEOUT_SEC:
                    await asyncio.sleep(settings.RUNPOD_POLL_INTERVAL_SEC)
                    elapsed += settings.RUNPOD_POLL_INTERVAL_SEC

                    poll_resp = await client.get(status_url, headers=headers)
                    poll_resp.raise_for_status()
                    poll_data = poll_resp.json()
                    status = poll_data.get("status")

                    print(f"[Pronunciation] RunPod job {job_id} status={status} ({elapsed:.0f}s)")

                    if status == "COMPLETED":
                        output = poll_data.get("output")
                        break
                    if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
                        raise RuntimeError(f"RunPod job {job_id} ended with status={status}: {poll_data}")
                else:
                    raise TimeoutError(f"RunPod job {job_id} did not complete within {settings.RUNPOD_POLL_TIMEOUT_SEC}s")

            if isinstance(output, dict) and output.get("error"):
                raise RuntimeError(f"RunPod handler returned an error: {output['error']}")

            result = analyze_gop_result(output)
            print(f"[Pronunciation] {result['total_phonemes']} phonemes, "
                  f"avg={result['utterance_avg']}, severe={result['distribution']['severe']}, "
                  f"worst={result['worst_phoneme']}")
            return result

    except Exception as e:
        print(f"[Pronunciation] RunPod call failed ({type(e).__name__}: {e}), returning empty result")
        return _DEFAULT_RESULT