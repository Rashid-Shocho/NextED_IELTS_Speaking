"""
Migration: adds the pronunciation_words table (Section 11 of the architecture doc).

One row per word the pronunciation model (wav2vec2-lv-60-espeak-cv-ft +
forced_align, see Appendix C) flags with a phoneme-level score. This
replaces the old aggregate ssl_ft_pron metrics and is what powers
longitudinal "what does this candidate keep mispronouncing" queries later
via a plain GROUP BY -- no vector search needed (Appendix A).

Does NOT touch speaking_sessions / speaking_parts / speaking_reports.
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

            cur.execute("""
                CREATE TABLE IF NOT EXISTS pronunciation_words (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id      UUID NOT NULL REFERENCES speaking_sessions(id) ON DELETE CASCADE,
                    part_id         UUID NOT NULL REFERENCES speaking_parts(id) ON DELETE CASCADE,
                    part_number     INT NOT NULL CHECK (part_number BETWEEN 1 AND 3),
                    user_id         TEXT,
                    word            TEXT NOT NULL,
                    phoneme         TEXT,
                    score           NUMERIC(4,3),           -- 0.000-1.000, wav2vec2/forced_align confidence
                    at_seconds      NUMERIC(8,3),            -- position in the audio
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
            print("✓ pronunciation_words")

            cur.execute("""
                CREATE INDEX IF NOT EXISTS pronunciation_words_session_idx
                ON pronunciation_words (session_id);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS pronunciation_words_user_word_idx
                ON pronunciation_words (user_id, word);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS pronunciation_words_low_score_idx
                ON pronunciation_words (score)
                WHERE score < 0.75;
            """)
            print("✓ indexes (including partial index for flagged/low-score words)")

            conn.commit()
            print("\npronunciation_words table created successfully!")


if __name__ == "__main__":
    migrate()