import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not configured")


def get_connection():
    return psycopg.connect(DATABASE_URL)


def create_speaking_tables():
    with get_connection() as conn:
        with conn.cursor() as cur:
            print("Connected to database successfully.")

            # 1. Enable pgvector extension
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            print("✓ vector extension ready")

            # 2. Speaking sessions
            cur.execute("""
                CREATE TABLE IF NOT EXISTS speaking_sessions (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    user_id         TEXT,                       -- links to your existing users.id if you want
                    status          TEXT NOT NULL DEFAULT 'pending',
                    -- pending | processing | completed | failed | needs_rerecording
                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
                    completed_at    TIMESTAMPTZ,
                    error_message   TEXT
                );
            """)
            print("✓ speaking_sessions")

            # 3. Individual parts (Part 1, 2, 3)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS speaking_parts (
                    id                  UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id          UUID NOT NULL REFERENCES speaking_sessions(id) ON DELETE CASCADE,
                    part_number         INT NOT NULL CHECK (part_number BETWEEN 1 AND 3),
                    question_text       TEXT NOT NULL,
                    audio_url           TEXT NOT NULL,          -- path or S3/MinIO URL
                    transcript          TEXT,
                    fluency_features    JSONB,
                    pronunciation       JSONB,                  -- output from ssl_ft_pron
                    status              TEXT NOT NULL DEFAULT 'pending'
                                        CHECK (status IN (
                                            'pending',
                                            'transcribing',
                                            'no_speech_detected',
                                            'pronunciation_done',
                                            'failed'
                                        )),
                    error_reason        TEXT,                   -- e.g. why status = 'failed'/'no_speech_detected'
                    created_at          TIMESTAMPTZ NOT NULL DEFAULT now(),
                    UNIQUE (session_id, part_number)
                );
            """)
            print("✓ speaking_parts")

            # 4. Final reports + embedding for pgvector
            #    Change 1536 → the real dimension of the embedding model you will use
            cur.execute("""
                CREATE TABLE IF NOT EXISTS speaking_reports (
                    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
                    session_id      UUID NOT NULL UNIQUE REFERENCES speaking_sessions(id) ON DELETE CASCADE,

                    -- Band scores
                    fluency         NUMERIC(3,1),
                    lexical         NUMERIC(3,1),
                    grammar         NUMERIC(3,1),
                    pronunciation   NUMERIC(3,1),
                    overall         NUMERIC(3,1),

                    evidence        JSONB,                      -- full structured output from the LLM
                    embedding       vector(1536),               -- adjust dimension later if needed

                    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
            print("✓ speaking_reports")

            # 5. HNSW index for fast similarity search
            cur.execute("""
                CREATE INDEX IF NOT EXISTS speaking_reports_embedding_idx
                ON speaking_reports
                USING hnsw (embedding vector_cosine_ops);
            """)
            print("✓ HNSW index on embedding")

            # 6. Helpful indexes
            cur.execute("""
                CREATE INDEX IF NOT EXISTS speaking_sessions_user_id_idx
                ON speaking_sessions (user_id);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS speaking_sessions_status_idx
                ON speaking_sessions (status);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS speaking_parts_session_id_idx
                ON speaking_parts (session_id);
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS speaking_parts_status_idx
                ON speaking_parts (status);
            """)
            print("✓ extra indexes")

            conn.commit()
            print("\nAll speaking tables created successfully!")


if __name__ == "__main__":
    create_speaking_tables()