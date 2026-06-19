"""ORM models.

Sign metadata + SBERT embeddings live in Postgres (embeddings as a pgvector
column with an HNSW cosine index). Raw MediaPipe landmark sequences are NOT
stored in the database — they live as compressed .npz files on disk, keyed by
sign id (see landmark_store.py). Postgres holds only the file path.
"""
from __future__ import annotations

from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    Boolean,
    DateTime,
    Index,
    Integer,
    String,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


# all-MiniLM-L6-v2 produces 384-dimensional sentence embeddings.
EMBEDDING_DIM = 384


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    full_name: Mapped[str | None] = mapped_column(String, nullable=True)
    email: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    hashed_password: Mapped[str] = mapped_column(String, nullable=False)
    role: Mapped[str] = mapped_column(String, default="Speaker")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)


class RefreshToken(Base):
    """Server-side refresh-token registry for rotation + revocation.

    Populated in the auth-hardening phase. A token is valid only if its jti is
    present here and revoked is false; logout / rotation flips revoked.
    """

    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    jti: Mapped[str] = mapped_column(String, unique=True, index=True, nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, index=True, nullable=False)
    revoked: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())


class Sign(Base):
    """A single sign: metadata + embedding. Landmarks are on disk."""

    __tablename__ = "signs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    word: Mapped[str] = mapped_column(String, nullable=False, index=True)
    # 'en' (ASL / signs.db) or 'ar' (ArSL / signs_ar.db)
    language: Mapped[str] = mapped_column(String, nullable=False, index=True)
    video_filename: Mapped[str | None] = mapped_column(String, nullable=True)
    landmark_file: Mapped[str | None] = mapped_column(String, nullable=True)
    frame_count: Mapped[int] = mapped_column(Integer, default=0)
    embedding: Mapped[list[float]] = mapped_column(Vector(EMBEDDING_DIM), nullable=True)

    __table_args__ = (
        UniqueConstraint("word", "language", name="uq_signs_word_language"),
        # HNSW index for fast approximate cosine-distance search.
        Index(
            "ix_signs_embedding_hnsw",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )
