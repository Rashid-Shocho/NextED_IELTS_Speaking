"""
Creates `speaking_training_samples` — a standalone table for building the
future IELTS speaking-scoring training dataset.

This does NOT modify or touch any existing tables
(speaking_sessions / speaking_parts / speaking_reports stay exactly as they are).

Each row = one audio response + question, with a Whisper transcript and an
expert examiner's IELTS band scores. Once a row is `annotation_status =
'verified'`, it's ready to be used to train/fine-tune a model that predicts
IELTS speaking band scores directly from audio + transcript (no RAG).
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


def create_training_data_table():
    with get_connection() as conn:
        with conn.cursor() as cur:
            print("Connected to database successfully.")

            cur.execute("""
                CREATE TABLE IF NOT EXISTS speaking_training_samples (
                    id                      UUID PRIMARY KEY DEFAULT gen_random_uuid(),

                    -- Who / where this sample came from
                    user_id                 TEXT NOT NULL,
                    session_id              UUID REFERENCES speaking_sessions(id) ON DELETE SET NULL,
                    part_id                 UUID REFERENCES speaking_parts(id) ON DELETE SET NULL,
                    part_number             INT CHECK (part_number BETWEEN 1 AND 3),

                    -- The prompt + recording
                    question_text           TEXT NOT NULL,
                    audio_url               TEXT NOT NULL,          -- Cloudflare (R2/Stream) URL
                    audio_duration_seconds  NUMERIC(6,2),
                    audio_format            TEXT,                   -- m4a, wav, mp3...

                    -- ASR transcript (from Whisper)
                    transcript_text         TEXT,
                    transcript_source       TEXT NOT NULL DEFAULT 'whisper'
                                             CHECK (transcript_source IN ('whisper', 'human', 'corrected')),
                    transcript_confidence   NUMERIC(4,3),           -- e.g. avg logprob / confidence score

                    -- Ground-truth labels: expert (human examiner) scores
                    expert_fluency          NUMERIC(3,1),
                    expert_lexical          NUMERIC(3,1),
                    expert_grammar          NUMERIC(3,1),
                    expert_pronunciation    NUMERIC(3,1),
                    expert_overall          NUMERIC(3,1),
                    expert_feedback         JSONB,                  -- structured comments per criterion
                    evaluator_id            TEXT,                   -- examiner who scored this sample
                    evaluated_at            TIMESTAMPTZ,

                    -- Dataset management
                    annotation_status       TEXT NOT NULL DEFAULT 'pending'
                                             CHECK (annotation_status IN ('pending', 'in_review', 'verified', 'rejected')),
                    dataset_split           TEXT
                                             CHECK (dataset_split IN ('train', 'validation', 'test')),
                    consent_given           BOOLEAN NOT NULL DEFAULT false,  -- user consented to use for training
                    is_active               BOOLEAN NOT NULL DEFAULT true,  -- soft-exclude bad samples
                    quality_notes           TEXT,                   -- e.g. "background noise", "cut off"

                    created_at              TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at              TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
            print("✓ speaking_training_samples")

            # Helpful indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS speaking_training_samples_user_id_idx
                ON speaking_training_samples (user_id);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS speaking_training_samples_split_idx
                ON speaking_training_samples (dataset_split);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS speaking_training_samples_status_idx
                ON speaking_training_samples (annotation_status);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS speaking_training_samples_session_id_idx
                ON speaking_training_samples (session_id);
            """)
            print("✓ indexes")

            # Auto-update `updated_at` on row changes
            cur.execute("""
                CREATE OR REPLACE FUNCTION set_updated_at()
                RETURNS TRIGGER AS $$
                BEGIN
                    NEW.updated_at = now();
                    RETURN NEW;
                END;
                $$ LANGUAGE plpgsql;
            """)
            cur.execute("""
                DROP TRIGGER IF EXISTS trg_speaking_training_samples_updated_at
                ON speaking_training_samples;
            """)
            cur.execute("""
                CREATE TRIGGER trg_speaking_training_samples_updated_at
                BEFORE UPDATE ON speaking_training_samples
                FOR EACH ROW EXECUTE FUNCTION set_updated_at();
            """)
            print("✓ updated_at trigger")

            conn.commit()
            print("\nspeaking_training_samples table created successfully!")
            print("Existing tables (speaking_sessions, speaking_parts, speaking_reports) were not modified.")


if __name__ == "__main__":
    create_training_data_table()