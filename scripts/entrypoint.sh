#!/bin/sh
# Container entrypoint: apply schema migrations, ensure sign data is loaded,
# then launch the server. Postgres readiness is guaranteed by the compose
# healthcheck (depends_on: service_healthy).
set -e

echo "[entrypoint] Applying Alembic migrations..."
alembic upgrade head

# Populate Postgres + the on-disk landmark store from the bundled SQLite DBs.
# The migration is idempotent (upsert by word+language); it is a no-op-ish
# refresh on subsequent boots.
echo "[entrypoint] Ensuring sign data is migrated..."
python scripts/migrate_to_postgres.py || echo "[entrypoint] WARNING: data migration step failed (continuing)"

echo "[entrypoint] Starting server..."
exec python app/server/main.py
