from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    DATABASE_URL: str
    GROQ_API_KEY: str
    HF_TOKEN: str

    # Shared secret the Next.js app sends as `X-Internal-Api-Key` on every
    # request. This service has no other auth (CORS is wildcard) so this is
    # the only thing stopping a random caller from burning Groq/RunPod quota.
    INTERNAL_API_KEY: str

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
    # With up to 7 segments now (vs 3 before), firing every pronunciation
    # call at once floods RunPod's limited worker pool and most requests
    # queue behind each other past any reasonable client timeout. Caps how
    # many segments' RunPod calls run at once; the rest wait their turn.
    RUNPOD_MAX_CONCURRENT: int = 3
    # RunPod's deployed handler hard-rejects audio over MAX_AUDIO_SECONDS=100
    # (returns status=FAILED, not a graceful trim). Part 2 allows up to 120s
    # of speech, which can exceed that. Real fix is raising the cap on the
    # RunPod worker itself and redeploying; this is a client-side safety net
    # so a submission never silently loses all pronunciation data over it in
    # the meantime. Kept a few seconds under 100 for encoding overhead.
    RUNPOD_MAX_AUDIO_SECONDS: float = 95.0

    # Cloudflare R2 (S3-compatible) -- archives every submitted audio clip
    # for the future training dataset (speaking_training_samples). All
    # optional: if unset, uploads are skipped and the pipeline behaves
    # exactly as before (local-disk-only), so this can be turned on
    # whenever R2 credentials are ready without touching code.
    R2_ACCOUNT_ID: str = ""
    R2_ACCESS_KEY_ID: str = ""
    R2_SECRET_ACCESS_KEY: str = ""
    R2_BUCKET_NAME: str = ""
    # Public base URL for reading objects back -- either your R2 custom
    # domain, or the bucket's public r2.dev URL if you enabled it (Cloudflare
    # dashboard -> R2 -> bucket -> Settings -> Public Access). No trailing
    # slash, e.g. "https://audio.yourdomain.com" or
    # "https://pub-xxxx.r2.dev".
    R2_PUBLIC_BASE_URL: str = ""

    class Config:
        env_file = ".env"
        extra = "ignore"

    @property
    def r2_configured(self) -> bool:
        return bool(
            self.R2_ACCOUNT_ID and self.R2_ACCESS_KEY_ID
            and self.R2_SECRET_ACCESS_KEY and self.R2_BUCKET_NAME
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()