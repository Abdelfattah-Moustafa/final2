"""Idempotent one-time migration: SQLite -> PostgreSQL + on-disk landmark store.

Reads the legacy SQLite sign databases (signs.db for English, signs_ar.db for
Arabic), extracts each sign's MediaPipe landmark sequence to a compressed .npz
file on disk, and loads the metadata + SBERT embedding into Postgres (embedding
as a pgvector column).

Safe to re-run: signs are upserted by (word, language) and landmark files are
overwritten in place. Verifies source vs destination row counts before exiting
non-zero on any mismatch.

Usage:
    python scripts/migrate_to_postgres.py
    DATABASE_URL=... python scripts/migrate_to_postgres.py --en-db data/signs.db
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sqlite3
import sys

import numpy as np

# Make app/server importable regardless of CWD.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app", "server"))

from db.base import SessionLocal, init_db  # noqa: E402
from db.models import EMBEDDING_DIM  # noqa: E402
from db.repository import SignRepository  # noqa: E402
import landmark_store  # noqa: E402


def reassemble_if_needed(db_path: str) -> None:
    """Reassemble signs.db from .parta/.partb parts if the DB is missing."""
    if os.path.exists(db_path):
        return
    part_a, part_b = db_path + ".parta", db_path + ".partb"
    if os.path.exists(part_a) and os.path.exists(part_b):
        print(f"  reassembling {db_path} from parts...")
        with open(db_path, "wb") as out:
            for p in (part_a, part_b):
                with open(p, "rb") as f:
                    shutil.copyfileobj(f, out)


def source_rows(db_path: str):
    """Yield (word, video_path, embedding ndarray, landmarks ndarray)."""
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        cur = conn.cursor()
        cur.execute(
            "SELECT word, video_path, embedding, landmarks FROM signs "
            "WHERE embedding IS NOT NULL AND landmarks IS NOT NULL"
        )
        for word, video_path, emb, lms in cur.fetchall():
            yield word, video_path, pickle.loads(emb), pickle.loads(lms)
    finally:
        conn.close()


def source_count(db_path: str) -> int:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        return conn.execute(
            "SELECT COUNT(*) FROM signs "
            "WHERE embedding IS NOT NULL AND landmarks IS NOT NULL"
        ).fetchone()[0]
    finally:
        conn.close()


def migrate_language(db_path: str, language: str) -> tuple[int, int]:
    """Migrate one SQLite DB into Postgres + npz store. Returns (src, dst) counts."""
    reassemble_if_needed(db_path)
    if not os.path.exists(db_path):
        print(f"  SKIP {language}: {db_path} not found")
        return 0, 0

    src = source_count(db_path)
    print(f"  {language}: {src} source rows in {db_path}")

    db = SessionLocal()
    repo = SignRepository(db)
    migrated = 0
    try:
        for word, video_path, emb, lms in source_rows(db_path):
            emb = np.asarray(emb, dtype=np.float32).ravel()
            if emb.shape[0] != EMBEDDING_DIM:
                raise ValueError(
                    f"{language}/{word}: embedding dim {emb.shape[0]} != {EMBEDDING_DIM}"
                )
            lms = np.asarray(lms)
            video_filename = os.path.basename(video_path) if video_path else None

            # Upsert metadata first to obtain a stable sign id, then key the
            # landmark file by that id.
            sign = repo.upsert(
                word=word,
                language=language,
                embedding=emb.tolist(),
                landmark_file=None,
                video_filename=video_filename,
                frame_count=int(lms.shape[0]) if lms.ndim >= 1 else 0,
            )
            db.flush()  # assigns sign.id
            rel = landmark_store.save(language, sign.id, lms)
            sign.landmark_file = rel
            migrated += 1
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

    db = SessionLocal()
    try:
        dst = SignRepository(db).count(language)
    finally:
        db.close()
    print(f"  {language}: migrated {migrated}, now {dst} rows in Postgres")
    return src, dst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--en-db", default=os.path.join(ROOT, "data", "signs.db"))
    ap.add_argument("--ar-db", default=os.path.join(ROOT, "data", "signs_ar.db"))
    args = ap.parse_args()

    print("Ensuring Postgres schema (pgvector extension + tables)...")
    init_db()

    ok = True
    for db_path, language in ((args.en_db, "en"), (args.ar_db, "ar")):
        src, dst = migrate_language(db_path, language)
        if src and dst != src:
            print(f"  MISMATCH {language}: source {src} != destination {dst}")
            ok = False

    print("\nMigration complete." if ok else "\nMigration FAILED (count mismatch).")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
