"""Repository / DAO layer.

Encapsulates all query logic so callers never write SQL or touch pgvector
operators directly. Vector similarity search runs as an INDEXED cosine-distance
query in Postgres (HNSW), replacing the old in-Python brute-force loop over all
embeddings.
"""
from __future__ import annotations

from typing import Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from .models import RefreshToken, Sign, User


class SignRepository:
    def __init__(self, db: Session):
        self.db = db

    def words(self, language: str) -> list[str]:
        """All words for a language (cheap; strings only)."""
        rows = self.db.execute(
            select(Sign.word).where(Sign.language == language)
        ).scalars().all()
        return list(rows)

    def count(self, language: str) -> int:
        return int(
            self.db.execute(
                select(func.count()).select_from(Sign).where(Sign.language == language)
            ).scalar_one()
        )

    def get_exact(self, word: str, language: str) -> Optional[Sign]:
        return self.db.execute(
            select(Sign).where(Sign.language == language, Sign.word == word)
        ).scalar_one_or_none()

    def nearest(
        self,
        embedding,
        language: str,
        max_distance: float,
    ) -> Optional[tuple[Sign, float]]:
        """Return (sign, cosine_distance) of the closest sign within max_distance.

        cosine_distance = 1 - cosine_similarity, so a similarity threshold of
        0.65 corresponds to max_distance = 0.35.
        """
        distance = Sign.embedding.cosine_distance(embedding).label("distance")
        row = self.db.execute(
            select(Sign, distance)
            .where(Sign.language == language, Sign.embedding.isnot(None))
            .order_by(distance)
            .limit(1)
        ).first()
        if row is None:
            return None
        sign, dist = row[0], float(row[1])
        if dist <= max_distance:
            return sign, dist
        return None



    def upsert(
        self,
        *,
        word: str,
        language: str,
        embedding,
        landmark_file: Optional[str],
        video_filename: Optional[str],
        frame_count: int,
    ) -> Sign:
        existing = self.get_exact(word, language)
        if existing is None:
            existing = Sign(word=word, language=language)
            self.db.add(existing)
        existing.embedding = embedding
        existing.landmark_file = landmark_file
        existing.video_filename = video_filename
        existing.frame_count = frame_count
        return existing


class UserRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_by_email(self, email: str) -> Optional[User]:
        return self.db.execute(
            select(User).where(func.lower(User.email) == email.lower())
        ).scalar_one_or_none()

    def create(self, *, full_name, email, hashed_password, role) -> User:
        user = User(
            full_name=full_name,
            email=email,
            hashed_password=hashed_password,
            role=role,
        )
        self.db.add(user)
        return user


class RefreshTokenRepository:
    def __init__(self, db: Session):
        self.db = db

    def add(self, *, jti: str, user_id: int, expires_at) -> RefreshToken:
        tok = RefreshToken(jti=jti, user_id=user_id, expires_at=expires_at)
        self.db.add(tok)
        return tok

    def get(self, jti: str) -> Optional[RefreshToken]:
        return self.db.execute(
            select(RefreshToken).where(RefreshToken.jti == jti)
        ).scalar_one_or_none()

    def revoke(self, jti: str) -> None:
        tok = self.get(jti)
        if tok:
            tok.revoked = True
