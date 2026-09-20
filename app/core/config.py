from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    DATABASE_URL: str
    GROQ_API_KEY: str
    HF_TOKEN: str

    # Dedicated key for the final_scoring pass (Pass D) -- this is the call
    # most likely to hit Groq's TPM rate limit, since it fires last, after
    # Pass A (text_analysis) and Pass B (pronunciation) have already spent
    # this minute's budget on the same key. Groq's rate limits are
    # org-scoped, not per-key, so this only helps if it's a key from a
    # *separate* Groq organization -- a second key on the same org shares
    # the same TPM bucket and does nothing. Optional: falls back to
    # GROQ_API_KEY (see groq_api_key_final_scoring below) if unset, so
    # this is safe to deploy before the second account/key exists.
    GROQ_API_KEY_FINAL_SCORING: str = ""

    # Shared secret the Next.js app sends as `X-Internal-Api-Key` on every
    # request. This service has no other auth (CORS is wildcard) so this is
    # the only thing stopping a random caller from burning Groq/RunPod quota.
    INTERNAL_API_KEY: str

    # Comma-separated list of origins allowed to call this service directly
    # from a browser (should just be the deployed Next.js app's origin(s) --
    # this is server-to-server otherwise). Defaults to localhost for local
    # dev; MUST be overridden in the deployed env or the beta frontend won't
    # be able to reach this service at all.
    ALLOWED_ORIGINS: str = "http://localhost:3000"

    # Models
    GROQ_WHISPER_MODEL: str = "whisper-large-v3-turbo"
    GROQ_LLM_MODEL: str = "openai/gpt-oss-20b"
    HF_EMBEDDING_MODEL: str = "BAAI/bge-base-en-v1.5"
    EMBEDDING_DIM: int = 768

    # Retry policy for transient failures against Groq (ASR) and RunPod
    # (pronunciation). Both services fail this many times before the part
    # is marked "failed" and surfaced to the user -- see asr.py /
    # pronunciation.py. Kept small: this is beta-safety against blips, not
    # a substitute for fixing a service that's actually down.
    TRANSCRIBE_MAX_ATTEMPTS: int = 3
    TRANSCRIBE_RETRY_BACKOFF_SEC: float = 2.0
    PRONUNCIATION_MAX_ATTEMPTS: int = 2
    PRONUNCIATION_RETRY_BACKOFF_SEC: float = 3.0

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

    @property
    def allowed_origins_list(self) -> list[str]:
        return [o.strip() for o in self.ALLOWED_ORIGINS.split(",") if o.strip()]

    @property
    def groq_api_key_final_scoring(self) -> str:
        return self.GROQ_API_KEY_FINAL_SCORING or self.GROQ_API_KEY


@lru_cache
def get_settings() -> Settings:
    return Settings()