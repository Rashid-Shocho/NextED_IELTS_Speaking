"""
Read-only inspection of the current database schema. Doesn't modify
anything. Run this and paste the output back so we can confirm what
actually needs to change vs. what's already in place.
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


def inspect():
    with get_connection() as conn:
        with conn.cursor() as cur:
            print("=" * 70)
            print("TABLES")
            print("=" * 70)
            cur.execute("""
                SELECT table_name
                FROM information_schema.tables
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = [r[0] for r in cur.fetchall()]
            for t in tables:
                print(f"  - {t}")

            for table in tables:
                print("\n" + "=" * 70)
                print(f"COLUMNS: {table}")
                print("=" * 70)
                cur.execute("""
                    SELECT column_name, data_type, is_nullable, column_default
                    FROM information_schema.columns
                    WHERE table_schema = 'public' AND table_name = %s
                    ORDER BY ordinal_position;
                """, (table,))
                for col_name, dtype, nullable, default in cur.fetchall():
                    print(f"  {col_name:<25} {dtype:<20} nullable={nullable:<5} default={default}")

            print("\n" + "=" * 70)
            print("CHECK CONSTRAINTS")
            print("=" * 70)
            cur.execute("""
                SELECT conrelid::regclass AS table_name, conname, pg_get_constraintdef(oid)
                FROM pg_constraint
                WHERE contype = 'c' AND connamespace = 'public'::regnamespace
                ORDER BY 1;
            """)
            for tbl, name, definition in cur.fetchall():
                print(f"  [{tbl}] {name}: {definition}")

            print("\n" + "=" * 70)
            print("INDEXES")
            print("=" * 70)
            cur.execute("""
                SELECT tablename, indexname, indexdef
                FROM pg_indexes
                WHERE schemaname = 'public'
                ORDER BY tablename, indexname;
            """)
            for tbl, name, definition in cur.fetchall():
                print(f"  [{tbl}] {name}")
                print(f"      {definition}")

            print("\n" + "=" * 70)
            print("EXTENSIONS")
            print("=" * 70)
            cur.execute("SELECT extname, extversion FROM pg_extension;")
            for name, version in cur.fetchall():
                print(f"  {name} ({version})")

            print("\n" + "=" * 70)
            print("ROW COUNTS")
            print("=" * 70)
            for t in tables:
                cur.execute(f'SELECT COUNT(*) FROM "{t}";')
                count = cur.fetchone()[0]
                print(f"  {t}: {count} rows")


if __name__ == "__main__":
    inspect()