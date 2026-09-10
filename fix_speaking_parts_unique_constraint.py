"""
One-time migration: speaking_parts had a UNIQUE (session_id, part_number)
constraint from before segments existed (back when there was exactly one
row per part). Now a part can have several segment rows (e.g. Part 1 =
intro + 3 topics), so that constraint is wrong and needs replacing with
UNIQUE (session_id, segment_id) instead.

Safe to re-run (IF EXISTS / IF NOT EXISTS).

Run once: python fix_speaking_parts_unique_constraint.py
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
                DROP CONSTRAINT IF EXISTS speaking_parts_session_id_part_number_key;
                """
            )
            cur.execute(
                """
                ALTER TABLE speaking_parts
                ADD CONSTRAINT speaking_parts_session_id_segment_id_key
                UNIQUE (session_id, segment_id);
                """
            )
            conn.commit()
            print("✓ Old (session_id, part_number) constraint dropped.")
            print("✓ New (session_id, segment_id) constraint added.")


if __name__ == "__main__":
    main()
