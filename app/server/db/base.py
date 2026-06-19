"""SQLAlchemy 2.x engine, session factory, and declarative base.

The database URL is externalized via the DATABASE_URL env var so the same
code runs against the docker-compose Postgres service, a locally installed
Postgres, or a test database. A connection pool is configured with
pre-ping so stale connections (e.g. after Postgres restarts) are recycled
transparently.
"""
from __future__ import annotations

import os

from sqlalchemy import create_engine, text
from sqlalchemy.orm import DeclarativeBase, sessionmaker

# Default points at the docker-compose / local dev Postgres. Override in .env.
DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql+psycopg://together:together@localhost:5432/together",
)

# pool_pre_ping recycles dead connections; pool_size/max_overflow keep a
# bounded pool so blocking ML work in a threadpool can't exhaust connections.
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=int(os.environ.get("DB_POOL_SIZE", "5")),
    max_overflow=int(os.environ.get("DB_MAX_OVERFLOW", "10")),
    future=True,
)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False, future=True)


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""


def get_db():
    """FastAPI dependency yielding a scoped Session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """Idempotently ensure the pgvector extension and all tables exist.

    Production schema changes go through Alembic (see alembic/); this helper
    exists so a fresh dev/test database boots cleanly without a manual step.
    """
    # Import models so they register on Base.metadata before create_all.
    from . import models  # noqa: F401

    with engine.begin() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
    Base.metadata.create_all(bind=engine)


