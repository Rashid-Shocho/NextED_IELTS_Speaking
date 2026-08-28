"""
Migration: adds per-part status tracking to the existing `speaking_parts`
table so the pipeline can implement the empty-audio / has_speech gate
(see simplified_flow.png):

    Whisper transcribes -> has_speech check ->
        no text  -> status = 'no_speech_detected', SKIP pronunciation + SKIP LLM
        has text -> ssl_ft_pron + fluency features -> status = 'pronunciation_done'

This only ALTERs the existing table (adds columns with safe defaults).
No existing rows, tables, or data are touched/dropped.
"""

import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured")


def get_connection():
    return psycopg.connect(DATABASE_URL)


def migrate():
    with get_connection() as conn:
        with conn.cursor() as cur:
            print("Connected to database successfully.")

            # 1. status column — tracks per-part pipeline progress
            cur.execute("""
                ALTER TABLE speaking_parts
                ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'pending';
            """)
            print("✓ speaking_parts.status added")

            # 2. Guard the allowed values (drop+recreate so this script is
            #    safely re-runnable if you ever add a new status later)
            cur.execute("""
                ALTER TABLE speaking_parts
                DROP CONSTRAINT IF EXISTS speaking_parts_status_check;
            """)
            cur.execute("""
                ALTER TABLE speaking_parts
                ADD CONSTRAINT speaking_parts_status_check
                CHECK (status IN (
                    'pending',              -- audio uploaded, not processed yet
                    'transcribing',         -- Whisper in progress
                    'no_speech_detected',   -- empty transcript -> user must re-record
                    'pronunciation_done',   -- transcript + pronunciation + fluency stored
                    'failed'                -- unexpected error during processing
                ));
            """)
            print("✓ status CHECK constraint")

            # 3. error_reason — short machine-readable reason, e.g. 'no_speech_detected'
            #    or an exception message, surfaced to the client so it knows why a
            #    part needs to be re-recorded.
            cur.execute("""
                ALTER TABLE speaking_parts
                ADD COLUMN IF NOT EXISTS error_reason TEXT;
            """)
            print("✓ speaking_parts.error_reason added")

            # 4. Helpful index for "which parts still need attention" queries
            cur.execute("""
                CREATE INDEX IF NOT EXISTS speaking_parts_status_idx
                ON speaking_parts (status);
            """)
            print("✓ index on status")

            conn.commit()
            print("\nMigration complete. Existing rows default to status = 'pending'.")


if __name__ == "__main__":
    migrate()