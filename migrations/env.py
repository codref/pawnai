"""Alembic environment configuration.

Reads the database DSN from (in priority order):
  1. ``paths.db_dsn`` in ``.pawnai.yml`` (project config file)
  2. ``DATABASE_URL`` environment variable
  3. Hard-coded default (``postgresql+psycopg://postgres:postgres@localhost:5432/pawnai``)

Wires the SQLAlchemy metadata from ``pawnai.core.database.Base`` so that
``alembic revision --autogenerate`` can diff the current schema against the
ORM models.
"""

from __future__ import annotations

from logging.config import fileConfig

from sqlalchemy import engine_from_config, pool

from alembic import context

# ── Load pawnai ORM metadata ──────────────────────────────────────────────────
# Import Base (and all models, so they register with metadata) before Alembic
# inspects target_metadata.
from pawnai.core.database import Base  # noqa: F401 – registers all ORM models
from pawnai.core.config import AppConfig

target_metadata = Base.metadata

# ── Alembic config object ─────────────────────────────────────────────────────
config = context.config

# Honour the logging config in alembic.ini
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# ── DSN resolution ────────────────────────────────────────────────────────────
# AppConfig reads .pawnai.yml (paths.db_dsn) and falls back to DATABASE_URL
# then to the hard-coded default — so a single source of truth for all tooling.
_dsn: str = AppConfig().get("db_dsn")
config.set_main_option("sqlalchemy.url", _dsn)


# ── Migration runners ─────────────────────────────────────────────────────────

def run_migrations_offline() -> None:
    """Emit SQL to stdout without a live DB connection (for SQL script output).

    Usage::

        alembic upgrade head --sql > schema.sql
    """
    context.configure(
        url=_dsn,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations against a live database connection."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
