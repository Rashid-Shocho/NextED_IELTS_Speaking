"""
One-time migration: adds `cloud_audio_url` to speaking_parts.

Does NOT touch speaking_sessions, speaking_reports, or
speaking_training_samples. Safe to re-run (IF NOT EXISTS).

Run once: python add_cloud_audio_url_column.py
"""
import os

import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured")


def main():
    with psycopg.connect(DATABASE_URL) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                ALTER TABLE speaking_parts
                ADD COLUMN IF NOT EXISTS cloud_audio_url TEXT;
                """
            )
            conn.commit()
            print("✓ speaking_parts.cloud_audio_url ready (added if it wasn't already there)")


if __name__ == "__main__":
    main()
