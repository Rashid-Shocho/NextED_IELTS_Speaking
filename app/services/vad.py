"""
Silero VAD pass -- speech/silence segments, deliberately NOT Whisper's
word-level timestamps (see architecture doc, Section 4).

Decodes audio via the ffmpeg binary directly (subprocess) instead of
torchaudio.load(), which on recent torchaudio versions requires the
separate `torchcodec` package and is fragile to set up on Windows.
Uses settings.FFMPEG_PATH so it works whether ffmpeg is on PATH (Linux/
RunPod) or only reachable via an explicit path (Windows dev, e.g. via
ffmpeg-downloader).

Runs on the CPU-only orchestrator, not the RunPod GPU worker -- it's
signal processing, not a model-scoring call.

Fail-soft: if ffmpeg or Silero VAD aren't available, falls back to
assuming speech is present rather than blocking the pipeline in dev.
"""

import tempfile
import os
import subprocess
from pathlib import Path

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
    ffmpeg: any input format (m4a/mp3/wav/webm/opus) -> raw PCM16LE,
    mono, 16kHz. Written to a temp file first (not piped via stdin) --
    MP4/M4A containers can store their metadata (moov atom) at the end
    of the file, which requires a seekable input; a pipe isn't seekable
    and causes ffmpeg to fail with "partial file" / "Invalid data".
    """
    with tempfile.NamedTemporaryFile(suffix=".m4a", delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name

    try:
        proc = subprocess.run(
            [
                settings.FFMPEG_PATH,
                "-nostdin",
                "-hide_banner",
                "-loglevel", "warning",
                "-i", tmp_path,
                "-map", "0:a:0",
                "-vn",
                "-f", "s16le",
                "-acodec", "pcm_s16le",
                "-ac", "1",
                "-ar", str(TARGET_SR),
                "pipe:1",
            ],
            capture_output=True,
        )
    finally:
        os.unlink(tmp_path)

    stderr_text = proc.stderr.decode(errors="ignore") if proc.stderr else ""

    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg exited {proc.returncode}: {stderr_text[-800:]}")

    if not proc.stdout:
        raise RuntimeError(f"ffmpeg produced no output (exit 0). stderr: {stderr_text[-800:]}")

    pcm = np.frombuffer(proc.stdout, dtype=np.int16)
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
            raise ValueError("ffmpeg produced 0 audio samples (decode failed or empty file)")

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

    except subprocess.CalledProcessError as e:
        stderr = e.stderr.decode(errors="ignore") if e.stderr else ""
        print(f"[VAD] ffmpeg decode failed: {stderr[-500:]}")
        print("[VAD] Falling back to assuming speech is present (dev fallback).")
        return {
            "has_speech": True,
            "segments": [{"start": 0.0, "end": 4.0}],
            "total_duration_sec": 4.0,
        }

    except Exception as e:
        print(f"[VAD] Error: {type(e).__name__}: {e}")
        print("[VAD] Falling back to assuming speech is present (dev fallback).")
        return {
            "has_speech": True,
            "segments": [{"start": 0.0, "end": 4.0}],
            "total_duration_sec": 4.0,
        }