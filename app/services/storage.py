"""
Uploads audio clips to Cloudflare R2 (S3-compatible) so they're durably
archived somewhere other than local disk -- feeds speaking_training_samples
for the future fine-tuning dataset.

Deliberately fail-soft: if R2 isn't configured (see Settings.r2_configured)
or the upload errors for any reason, this returns None rather than raising.
Local-disk processing (VAD/ASR/pronunciation) never depends on this --
losing R2 should never break a live evaluation.
"""

import mimetypes
from pathlib import Path

import boto3
from botocore.config import Config as BotoConfig

from app.core.config import get_settings

settings = get_settings()

_client = None


def _get_client():
    global _client
    if _client is not None:
        return _client
    _client = boto3.client(
        "s3",
        endpoint_url=f"https://{settings.R2_ACCOUNT_ID}.r2.cloudflarestorage.com",
        aws_access_key_id=settings.R2_ACCESS_KEY_ID,
        aws_secret_access_key=settings.R2_SECRET_ACCESS_KEY,
        config=BotoConfig(signature_version="s3v4"),
        region_name="auto",
    )
    return _client


def upload_audio_to_r2(local_path: Path, key: str) -> str | None:
    """
    Synchronous (boto3 has no native async client) -- call via
    asyncio.to_thread from async code. Returns the public URL on success,
    or None if R2 isn't configured or the upload fails for any reason.
    """
    if not settings.r2_configured:
        return None

    content_type = mimetypes.guess_type(str(local_path))[0] or "application/octet-stream"

    try:
        client = _get_client()
        client.upload_file(
            str(local_path),
            settings.R2_BUCKET_NAME,
            key,
            ExtraArgs={"ContentType": content_type},
        )
    except Exception as e:
        print(f"[Storage] R2 upload failed for {local_path} -> {key}: {type(e).__name__}: {e}")
        return None

    if not settings.R2_PUBLIC_BASE_URL:
        print(
            f"[Storage] Uploaded to R2 as '{key}' but R2_PUBLIC_BASE_URL is not set -- "
            "cannot build a public URL. Set it in .env (custom domain or r2.dev URL)."
        )
        return None

    return f"{settings.R2_PUBLIC_BASE_URL.rstrip('/')}/{key}"
