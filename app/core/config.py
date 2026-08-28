from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    DATABASE_URL: str
    GROQ_API_KEY: str
    HF_TOKEN: str

    # Models
    GROQ_WHISPER_MODEL: str = "whisper-large-v3-turbo"
    GROQ_LLM_MODEL: str = "openai/gpt-oss-20b"
    HF_EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIM: int = 768

    # ffmpeg (see app/services/vad.py)
    FFMPEG_PATH: str = "ffmpeg"

    # RunPod pronunciation endpoint (wav2vec2-lv-60-espeak-cv-ft + forced_align)
    RUNPOD_API_KEY: str
    RUNPOD_PRONUNCIATION_URL: str = "https://api.runpod.ai/v2/xleyv5ilspgpbp/runsync"
    RUNPOD_POLL_INTERVAL_SEC: float = 2.0
    RUNPOD_POLL_TIMEOUT_SEC: float = 120.0

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()