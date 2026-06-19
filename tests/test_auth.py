"""Unit tests for auth utilities (no DB / Postgres required)."""
import os
import sys
import time

# Point to the server package so imports resolve.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "app", "server"))

import pytest

# Set a deterministic secret before importing auth so tests are reproducible.
os.environ.setdefault("JWT_SECRET_KEY", "test-secret-key-for-unit-tests")
os.environ.setdefault("ACCESS_TOKEN_EXPIRE_MINUTES", "60")
os.environ.setdefault("REFRESH_TOKEN_EXPIRE_DAYS", "7")

from auth import (
    _SlidingWindowLimiter,
    _ph,
    create_access_token,
    decode_token,
    get_password_hash,
    needs_rehash,
    verify_password,
)
from fastapi import HTTPException


# ── Password hashing ──────────────────────────────────────────

def test_argon2_roundtrip():
    h = get_password_hash("s3cr3tP@ssword!")
    assert h.startswith("$argon2id$")
    assert verify_password("s3cr3tP@ssword!", h) is True


def test_wrong_password_fails():
    h = get_password_hash("correct-horse")
    assert verify_password("wrong-horse", h) is False


def test_bcrypt_legacy_verify():
    import bcrypt
    legacy = bcrypt.hashpw(b"oldpass", bcrypt.gensalt()).decode()
    assert verify_password("oldpass", legacy) is True
    assert verify_password("badpass", legacy) is False


def test_needs_rehash_bcrypt():
    import bcrypt
    legacy = bcrypt.hashpw(b"x", bcrypt.gensalt()).decode()
    assert needs_rehash(legacy) is True


def test_needs_rehash_argon2():
    h = get_password_hash("x")
    assert needs_rehash(h) is False


# ── Access token ──────────────────────────────────────────────

def test_access_token_roundtrip():
    tok = create_access_token({"sub": "user@example.com", "user_id": 42})
    payload = decode_token(tok)
    assert payload["sub"] == "user@example.com"
    assert payload["user_id"] == 42
    assert payload["type"] == "access"


def test_expired_access_token_raises():
    from datetime import timedelta
    tok = create_access_token({"sub": "x"}, expires_delta=timedelta(seconds=-1))
    with pytest.raises(HTTPException) as exc_info:
        decode_token(tok)
    assert exc_info.value.status_code == 401


def test_tampered_token_raises():
    tok = create_access_token({"sub": "legit@test.com"})
    parts = tok.split(".")
    # corrupt the signature
    tampered = parts[0] + "." + parts[1] + ".badsignature"
    with pytest.raises(HTTPException) as exc_info:
        decode_token(tampered)
    assert exc_info.value.status_code == 401


# ── Rate limiter ──────────────────────────────────────────────

def test_rate_limiter_allows_under_limit():
    lim = _SlidingWindowLimiter(max_requests=5, window_seconds=60)
    for _ in range(5):
        assert lim.allow("ip1") is True


def test_rate_limiter_blocks_over_limit():
    lim = _SlidingWindowLimiter(max_requests=3, window_seconds=60)
    for _ in range(3):
        lim.allow("ip2")
    assert lim.allow("ip2") is False


def test_rate_limiter_per_ip_isolation():
    lim = _SlidingWindowLimiter(max_requests=2, window_seconds=60)
    lim.allow("ipA")
    lim.allow("ipA")
    assert lim.allow("ipA") is False
    # ipB is a separate key
    assert lim.allow("ipB") is True


def test_rate_limiter_window_resets():
    lim = _SlidingWindowLimiter(max_requests=2, window_seconds=1)
    lim.allow("ipC")
    lim.allow("ipC")
    assert lim.allow("ipC") is False
    time.sleep(1.1)
    assert lim.allow("ipC") is True
