"""Backward-compatibility shim.

The data layer now lives in the ``db`` package (SQLAlchemy 2.x + Postgres).
Existing imports like ``from database import engine, Base, User, get_db`` keep
working by re-exporting from there.
"""
from db.base import Base, engine, SessionLocal, get_db, init_db  # noqa: F401
from db.models import User, Sign, RefreshToken  # noqa: F401

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "init_db",
    "User",
    "Sign",
    "RefreshToken",
]
