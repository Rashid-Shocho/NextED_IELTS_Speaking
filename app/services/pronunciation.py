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
import io
import re
import wave
from pathlib import Path

import av
import httpx

from app.core.config import get_settings
from app.services.pronunciation_parser import analyze_gop_result

settings = get_settings()

ALLOWED_AUDIO_FORMATS = {"wav", "mp3", "m4a", "ogg", "flac"}
TARGET_SR = 16000

# Caps how many segments' RunPod calls run at once (see RUNPOD_MAX_CONCURRENT).
# Created lazily so it binds to whatever event loop is actually running --
# a Semaphore created at import time can bind to the wrong loop in some
# ASGI server setups.
_runpod_semaphore: asyncio.Semaphore | None = None


def _get_runpod_semaphore() -> asyncio.Semaphore:
    global _runpod_semaphore
    if _runpod_semaphore is None:
        _runpod_semaphore = asyncio.Semaphore(settings.RUNPOD_MAX_CONCURRENT)
    return _runpod_semaphore


def _status_url(run_url: str, job_id: str) -> str:
    """
    RunPod's synchronous endpoint is at .../runsync, but there's no GET
    status route at that same path -- status checks go to .../status/{id}
    instead. The pattern must match BOTH /run and /runsync (RunPod supports
    either as the base path depending on how the endpoint was called);
    matching only /run left /runsync URLs completely unchanged, so polling
    silently GET-ed the POST-only /runsync route and always got a 404.
    """
    new_url, n = re.subn(r"/run(sync)?/?$", f"/status/{job_id}", run_url)
    if n == 0:
        raise ValueError(f"Could not derive a /status/ URL from RUNPOD_PRONUNCIATION_URL={run_url!r}")
    return new_url


_DEFAULT_RESULT = {
    "utterance_avg": None,
    "total_phonemes": 0,
    "distribution": {"excellent": 0, "good": 0, "moderate": 0, "severe": 0},
    "severe_flags": [],
    "worst_phoneme": None,
}


def _decode_trim_encode_wav(input_bytes: bytes, max_seconds: float) -> tuple[bytes, bool]:
    """
    Decodes any input format via PyAV, trims to max_seconds if longer (RunPod
    hard-rejects audio over its own MAX_AUDIO_SECONDS instead of trimming
    itself), and re-encodes as 16kHz mono WAV. Returns (wav_bytes, was_trimmed).
    """
    container = av.open(io.BytesIO(input_bytes))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SR)

    pcm_chunks = []
    try:
        for frame in container.decode(audio=0):
            for rframe in resampler.resample(frame):
                pcm_chunks.append(rframe.to_ndarray().tobytes())
        for rframe in resampler.resample(None):
            pcm_chunks.append(rframe.to_ndarray().tobytes())
    finally:
        container.close()

    if not pcm_chunks:
        raise RuntimeError("PyAV decoded 0 audio frames")

    pcm_bytes = b"".join(pcm_chunks)

    max_bytes = int(max_seconds * TARGET_SR) * 2  # 16-bit PCM = 2 bytes/sample
    was_trimmed = len(pcm_bytes) > max_bytes
    if was_trimmed:
        pcm_bytes = pcm_bytes[:max_bytes]

    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(TARGET_SR)
        wf.writeframes(pcm_bytes)

    return wav_buffer.getvalue(), was_trimmed


async def _load_audio_b64_and_format(audio_url: str) -> tuple[str, str]:
    """
    Reads audio (currently only local paths are handled) and returns
    (base64_string, format_extension). audio_url is still the parameter
    name for now since that's what the rest of the pipeline calls it --
    despite the name, it's treated as a local file path here.

    Always decodes+re-encodes to WAV via PyAV, regardless of the original
    format -- this guarantees RunPod's MAX_AUDIO_SECONDS cap is enforced
    client-side too (see settings.RUNPOD_MAX_AUDIO_SECONDS), rather than
    only handling the "wrong format" case and letting long clips through
    to fail server-side.

    TODO: once audio is hosted somewhere RunPod's workers can reach
    (S3/Cloudflare R2/presigned URL), send {"audio_url": ...} directly
    instead of reading+base64-encoding, and skip this function entirely.
    """
    local_path = Path(audio_url)
    if not local_path.is_absolute():
        local_path = Path.cwd() / audio_url
    if not local_path.exists():
        raise FileNotFoundError(f"Local audio file not found: {local_path}")

    raw_bytes = local_path.read_bytes()
    wav_bytes, was_trimmed = _decode_trim_encode_wav(raw_bytes, settings.RUNPOD_MAX_AUDIO_SECONDS)

    if was_trimmed:
        print(
            f"[Pronunciation] Audio exceeded {settings.RUNPOD_MAX_AUDIO_SECONDS}s cap, "
            f"trimmed before sending to RunPod (raise MAX_AUDIO_SECONDS on the RunPod "
            f"worker + redeploy to fix properly instead of trimming)"
        )

    audio_b64 = base64.b64encode(wav_bytes).decode("ascii")
    return audio_b64, "wav"


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

        async with _get_runpod_semaphore():
            # Long enough for RunPod to queue behind other concurrent
            # segments AND cold-start a worker AND process up to ~90s of
            # audio -- 60s was fine for one-off calls but not for several
            # segments' worth of concurrent load.
            async with httpx.AsyncClient(timeout=180.0) as client:
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