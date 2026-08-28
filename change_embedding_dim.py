import os
import psycopg
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

with psycopg.connect(DATABASE_URL) as conn:
    with conn.cursor() as cur:
        print("Changing embedding column to vector(768)...")
        cur.execute("ALTER TABLE speaking_reports DROP COLUMN IF EXISTS embedding;")
        cur.execute("ALTER TABLE speaking_reports ADD COLUMN embedding vector(768);")
        cur.execute("""
            CREATE INDEX IF NOT EXISTS speaking_reports_embedding_idx
            ON speaking_reports
            USING hnsw (embedding vector_cosine_ops);
        """)
        conn.commit()
        print("✓ Done – embedding column is now vector(768)")