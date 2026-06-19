"""Database package: SQLAlchemy 2.x engine, ORM models, and repositories.

Storage is abstracted behind this package so the rest of the app never
talks to the driver directly:

  - base.py        engine / session / Base / get_db / init_db
  - models.py      ORM models (User, Sign, RefreshToken)
  - repository.py  DAO layer (vector search, user lookups)
"""

from .base import Base, engine, SessionLocal, get_db, init_db, DATABASE_URL
from .models import User, Sign, RefreshToken

__all__ = [
    "Base",
    "engine",
    "SessionLocal",
    "get_db",
    "init_db",
    "DATABASE_URL",
    "User",
    "Sign",
    "RefreshToken",
]
