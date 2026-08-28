"""
Isolates the VAD call from the rest of the pipeline so the real exception
is visible immediately, instead of scrolling through uvicorn's log.
Run: python tests/test_vad_standalone.py
"""

import asyncio
from app.services.vad import detect_speech_segments


async def main():
    for path in ["audio_samples/audio1.m4a", "audio_samples/audio2.m4a", "audio_samples/audio3.m4a"]:
        print(f"\n--- {path} ---")
        result = await detect_speech_segments(path)
        print(result)


if __name__ == "__main__":
    asyncio.run(main())