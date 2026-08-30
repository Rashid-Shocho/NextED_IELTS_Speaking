"""
One-time migration: adds `segment_id` and `label` to speaking_parts.

Needed for the new variable-segment architecture (Part 1 = intro + 3 topic
cards, Part 3 = 2 topic cards, each its own row) instead of exactly one row
per part_number. Existing rows get a best-effort backfilled segment_id
derived from part_number so old data doesn't break -- backfilled rows won't
have a meaningful `label` beyond "Part N", which is fine since they're from
before this migration and won't be re-scored.

Safe to re-run (IF NOT EXISTS / idempotent backfill).

Run once: python add_segment_columns.py
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
                ADD COLUMN IF NOT EXISTS segment_id TEXT,
                ADD COLUMN IF NOT EXISTS label TEXT;
                """
            )
            # Backfill any pre-existing rows (from before this migration)
            # with a synthetic segment_id/label so NOT NULL-style code paths
            # downstream don't choke on legacy data.
            cur.execute(
                """
                UPDATE speaking_parts
                SET segment_id = COALESCE(segment_id, 'part' || part_number),
                    label = COALESCE(label, 'Part ' || part_number)
                WHERE segment_id IS NULL OR label IS NULL;
                """
            )
            conn.commit()
            print("✓ speaking_parts.segment_id and .label ready (added + backfilled if needed)")


if __name__ == "__main__":
    main()
