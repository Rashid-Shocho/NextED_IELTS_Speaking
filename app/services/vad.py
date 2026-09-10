"""
Silero VAD pass -- speech/silence segments, deliberately NOT Whisper's
word-level timestamps (see architecture doc, Section 4).

Decodes audio via PyAV (the `av` package), which bundles its own compiled
ffmpeg libraries inside the wheel -- no system ffmpeg binary or PATH
configuration required, which was a recurring pain point on Windows dev
machines (previously shelled out to a system `ffmpeg` binary via
subprocess, which depends on PATH/registry state being in sync with
whatever process Python is running in).

Runs on the CPU-only orchestrator, not the RunPod GPU worker -- it's
signal processing, not a model-scoring call.

Fail-soft: if PyAV or Silero VAD aren't available, falls back to
assuming speech is present rather than blocking the pipeline in dev.
"""

import io
from pathlib import Path

import av
import httpx
import numpy as np

from app.core.config import get_settings

settings = get_settings()

_vad_model = None
_get_speech_timestamps = None

TARGET_SR = 16000


def _load_vad():
    global _vad_model, _get_speech_timestamps
    if _vad_model is not None:
        return
    import torch

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    _vad_model = model
    _get_speech_timestamps = utils[0]  # utils = (get_speech_timestamps, ...)


async def _load_audio_bytes(audio_url: str) -> bytes:
    if not audio_url.startswith(("http://", "https://")):
        local_path = Path(audio_url)
        if not local_path.is_absolute():
            local_path = Path.cwd() / audio_url
        if not local_path.exists():
            raise FileNotFoundError(f"Local audio file not found: {local_path}")
        return local_path.read_bytes()

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True) as client:
        resp = await client.get(audio_url)
        resp.raise_for_status()
        return resp.content


def _decode_to_pcm16_mono_16k(audio_bytes: bytes) -> np.ndarray:
    """
    PyAV: any input container/codec (m4a/mp3/wav/webm+opus, etc.) -> PCM16
    mono, 16kHz float32 samples in [-1, 1]. Works directly on in-memory
    bytes -- no temp file, no subprocess, no system ffmpeg dependency.
    """
    container = av.open(io.BytesIO(audio_bytes))
    resampler = av.AudioResampler(format="s16", layout="mono", rate=TARGET_SR)

    chunks: list[np.ndarray] = []
    try:
        for frame in container.decode(audio=0):
            for rframe in resampler.resample(frame):
                chunks.append(rframe.to_ndarray())
        # flush any samples buffered inside the resampler
        for rframe in resampler.resample(None):
            chunks.append(rframe.to_ndarray())
    finally:
        container.close()

    if not chunks:
        raise RuntimeError("PyAV decoded 0 audio frames (empty/corrupt file, or unsupported codec)")

    pcm = np.concatenate(chunks, axis=1).reshape(-1)  # s16 mono ndarray is shape (1, N)
    return pcm.astype(np.float32) / 32768.0


async def detect_speech_segments(audio_url: str) -> dict:
    """
    Returns:
        {
          "has_speech": bool,
          "segments": [{"start": float, "end": float}, ...],  # seconds
          "total_duration_sec": float,
        }
    """
    try:
        import torch

        _load_vad()

        audio_bytes = await _load_audio_bytes(audio_url)
        samples = _decode_to_pcm16_mono_16k(audio_bytes)

        if samples.size == 0:
            raise ValueError("PyAV produced 0 audio samples (decode failed or empty file)")

        waveform = torch.from_numpy(samples)

        speech_timestamps = _get_speech_timestamps(
            waveform, _vad_model, sampling_rate=TARGET_SR
        )

        total_duration = len(samples) / TARGET_SR
        segments = [
            {"start": ts["start"] / TARGET_SR, "end": ts["end"] / TARGET_SR}
            for ts in speech_timestamps
        ]

        print(f"[VAD] {len(segments)} speech segment(s) in {total_duration:.2f}s audio")

        return {
            "has_speech": len(segments) > 0,
            "segments": segments,
            "total_duration_sec": round(total_duration, 2),
        }

    except Exception as e:
        print(f"[VAD] Error: {type(e).__name__}: {e}")
        print("[VAD] Falling back to assuming speech is present (dev fallback).")
        return {
            "has_speech": True,
            "segments": [{"start": 0.0, "end": 4.0}],
            "total_duration_sec": 4.0,
        }