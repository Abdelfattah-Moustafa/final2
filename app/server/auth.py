"""Authentication utilities: Argon2id hashing, JWT (access + refresh), rate-limiting."""
from __future__ import annotations

import collections
import os
import time as _time
import uuid
from datetime import datetime, timedelta
from threading import Lock
from typing import Optional

import bcrypt  # kept for verifying legacy hashes during transparent migration
from argon2 import PasswordHasher
from argon2.exceptions import VerifyMismatchError
from fastapi import Cookie, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt

# ── Config from environment ───────────────────────────────────
SECRET_KEY = os.environ.get("JWT_SECRET_KEY", "dev-insecure-change-me-in-production")
ALGORITHM = os.environ.get("JWT_ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.environ.get("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
REFRESH_TOKEN_EXPIRE_DAYS = int(os.environ.get("REFRESH_TOKEN_EXPIRE_DAYS", "7"))

# ── Argon2id hasher (OWASP recommended params) ────────────────
_ph = PasswordHasher(
    time_cost=3,
    memory_cost=65536,  # 64 MB
    parallelism=2,
    hash_len=32,
    salt_len=16,
)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login", auto_error=False)


# ── Password utilities ────────────────────────────────────────
def get_password_hash(password: str) -> str:
    return _ph.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    """Verify against Argon2id or legacy bcrypt (transparent upgrade path)."""
    if hashed.startswith(("$2b$", "$2a$", "$2y$")):
        try:
            return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
        except Exception:
            return False
    try:
        return _ph.verify(hashed, plain)
    except VerifyMismatchError:
        return False
    except Exception:
        return False


def needs_rehash(hashed: str) -> bool:
    """True if the stored hash uses a legacy scheme and should be upgraded to Argon2id."""
    return hashed.startswith(("$2b$", "$2a$", "$2y$"))


# ── Access token ──────────────────────────────────────────────
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token.",
            headers={"WWW-Authenticate": "Bearer"},
        )


# ── Refresh token ─────────────────────────────────────────────
def create_refresh_token(user_id: int, db) -> tuple[str, str]:
    """Register a refresh token in the DB and return (jti, signed_jwt).

    The caller is responsible for committing the DB session.
    """
    jti = str(uuid.uuid4())
    expires_at = datetime.utcnow() + timedelta(days=REFRESH_TOKEN_EXPIRE_DAYS)
    signed = jwt.encode(
        {"sub": str(user_id), "jti": jti, "exp": expires_at, "type": "refresh"},
        SECRET_KEY,
        algorithm=ALGORITHM,
    )
    from db.repository import RefreshTokenRepository
    RefreshTokenRepository(db).add(jti=jti, user_id=user_id, expires_at=expires_at)
    return jti, signed


def rotate_refresh_token(old_token_str: str, db) -> tuple[int, str]:
    """Validate the old refresh token, revoke it, issue a new one.

    Returns (user_id, new_signed_jwt). Raises HTTP 401 on any failure.
    """
    try:
        payload = jwt.decode(old_token_str, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token.")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Not a refresh token.")

    jti = payload.get("jti")
    user_id = int(payload["sub"])

    from db.repository import RefreshTokenRepository
    repo = RefreshTokenRepository(db)
    rt = repo.get(jti)
    if rt is None or rt.revoked:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token has been revoked.")
    if rt.expires_at < datetime.utcnow():
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token expired.")

    repo.revoke(jti)
    _, new_signed = create_refresh_token(user_id, db)
    return user_id, new_signed


# ── Current-user dependency ───────────────────────────────────
def get_current_user(bearer: Optional[str] = Depends(oauth2_scheme)) -> dict:
    """FastAPI dependency: validate JWT from the Authorization: Bearer header."""
    if not bearer:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    payload = decode_token(bearer)
    if payload.get("type") == "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token required, not refresh token.",
        )
    return payload


# ── Rate limiter (sliding window, in-memory) ──────────────────
class _SlidingWindowLimiter:
    def __init__(self, max_requests: int, window_seconds: int):
        self._max = max_requests
        self._window = window_seconds
        self._hits: dict[str, list[float]] = collections.defaultdict(list)
        self._lock = Lock()

    def allow(self, key: str) -> bool:
        now = _time.time()
        cutoff = now - self._window
        with self._lock:
            recent = [t for t in self._hits[key] if t > cutoff]
            if len(recent) >= self._max:
                self._hits[key] = recent
                return False
            recent.append(now)
            self._hits[key] = recent
            return True


_auth_limiter = _SlidingWindowLimiter(max_requests=10, window_seconds=60)


def require_auth_rate_limit(request: Request) -> None:
    """FastAPI dependency: 10 auth requests per 60 s per IP."""
    ip = request.client.host if request.client else "unknown"
    if not _auth_limiter.allow(ip):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many requests. Please wait a moment before trying again.",
            headers={"Retry-After": "60"},
        )
