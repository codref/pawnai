"""PostgreSQL database layer using SQLAlchemy ORM + pgvector.

This module defines the shared ORM models, engine factory, and session helper
used across the pawnai package.  All LanceDB operations have been replaced by
their PostgreSQL + pgvector equivalents here.

Tables
------
embeddings
    Stores per-segment speaker embeddings extracted by pyannote.audio.
    The ``embedding`` column is a pgvector ``vector(512)`` for cosine-similarity
    searches.

speaker_names
    Maps (audio_file, local_speaker_label) pairs to human-readable names
    assigned via ``pawnai label``.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime
from typing import Generator, Optional

from sqlalchemy import DateTime, Float, String, create_engine, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

from pgvector.sqlalchemy import Vector


# Dimension produced by pyannote/embedding model.
EMBEDDING_DIM = 512


# ──────────────────────────────────────────────────────────────────────────────
# ORM Models
# ──────────────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    pass


class Embedding(Base):
    """Per-segment speaker embedding record.

    Columns
    -------
    id:
        ``"<basename>_<local_label>_<idx>"`` – deterministic primary key so
        that re-running diarization on the same file is idempotent via INSERT
        OR IGNORE / conflict handling.
    audio_file:
        Absolute or relative path to the source audio file.
    local_speaker_label:
        Pyannote speaker label, e.g. ``"SPEAKER_00"``.
    start_time / end_time:
        Segment boundaries in seconds.
    embedding:
        512-dimensional float32 vector from pyannote/embedding.
    """

    __tablename__ = "embeddings"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    audio_file: Mapped[str] = mapped_column(String, nullable=False)
    local_speaker_label: Mapped[str] = mapped_column(String, nullable=False)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    embedding: Mapped[list] = mapped_column(Vector(EMBEDDING_DIM), nullable=False)


class SpeakerName(Base):
    """Human-readable name assigned to a speaker in a specific audio file.

    Columns
    -------
    id:
        ``"<basename>_<local_label>"`` – unique per (file, speaker) pair.
    audio_file:
        Absolute or relative path to the audio file.
    local_speaker_label:
        Pyannote speaker label, e.g. ``"SPEAKER_00"``.
    speaker_name:
        Human-readable name, e.g. ``"Alice"``.
    labeled_at:
        Timestamp when the label was created/updated (UTC).
    """

    __tablename__ = "speaker_names"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    audio_file: Mapped[str] = mapped_column(String, nullable=False)
    local_speaker_label: Mapped[str] = mapped_column(String, nullable=False)
    speaker_name: Mapped[str] = mapped_column(String, nullable=False)
    labeled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


# ──────────────────────────────────────────────────────────────────────────────
# Engine factory
# ──────────────────────────────────────────────────────────────────────────────


def get_engine(dsn: str):
    """Create a SQLAlchemy engine from a DSN string.

    Uses the ``postgresql+psycopg`` dialect (psycopg v3).

    Args:
        dsn: PostgreSQL DSN, e.g.
             ``"postgresql+psycopg://postgres:postgres@localhost:5432/pawnai"``

    Returns:
        A :class:`sqlalchemy.engine.Engine` instance.
    """
    return create_engine(dsn, pool_pre_ping=True)


def init_db(engine) -> None:
    """Ensure the pgvector extension and all ORM tables exist.

    Safe to call multiple times (uses ``CREATE EXTENSION IF NOT EXISTS`` and
    ``CREATE TABLE IF NOT EXISTS`` semantics via SQLAlchemy's ``checkfirst``).

    Args:
        engine: A SQLAlchemy engine connected to the target PostgreSQL database.
    """
    with engine.connect() as conn:
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
        conn.commit()
    Base.metadata.create_all(engine, checkfirst=True)


# ──────────────────────────────────────────────────────────────────────────────
# Session helper
# ──────────────────────────────────────────────────────────────────────────────


@contextmanager
def get_session(engine) -> Generator[Session, None, None]:
    """Yield a SQLAlchemy :class:`Session` that is automatically committed or
    rolled back on exit.

    Usage::

        with get_session(engine) as session:
            session.add(Embedding(...))

    Args:
        engine: A SQLAlchemy engine.

    Yields:
        An active :class:`sqlalchemy.orm.Session`.
    """
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
