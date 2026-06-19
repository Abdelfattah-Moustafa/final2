"""Alembic environment.

Reads DATABASE_URL from the environment and targets the app's Base metadata so
autogenerate works. Online mode only (we always have a live Postgres).
"""
from __future__ import annotations

import os
import sys

from alembic import context
from sqlalchemy import engine_from_config, pool

# Make app/server importable so we can pull in the ORM metadata.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app", "server"))

from db.base import Base, DATABASE_URL  # noqa: E402
from db import models  # noqa: F401,E402  (register models on Base.metadata)

config = context.config
config.set_main_option("sqlalchemy.url", os.environ.get("DATABASE_URL", DATABASE_URL))

target_metadata = Base.metadata


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


run_migrations_online()
