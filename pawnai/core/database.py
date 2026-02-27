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
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine, text
from sqlalchemy.dialects.postgresql import JSONB
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


class TranscriptionSegment(Base):
    """One diarized+transcribed segment stored for a session.

    Columns
    -------
    id:
        Deterministic PK: ``"<session_id>_<segment_index>"``.
    session_id:
        Human-readable session name or auto-generated UUID.
    audio_file:
        Source audio path for this segment.
    original_speaker_label:
        Raw pyannote label (``"SPEAKER_00"`` etc.) – never a user-assigned
        name.  ``NULL`` for standalone ``transcribe`` runs.
    start_time / end_time:
        Globally adjusted segment boundaries in seconds.
    text:
        Transcribed text.
    words:
        JSON array of ``{"word", "start", "end"}`` objects.
    segment_index:
        Zero-based position within the session.
    created_at:
        UTC timestamp when the row was written.
    """

    __tablename__ = "transcription_segments"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    audio_file: Mapped[str] = mapped_column(String, nullable=False)
    original_speaker_label: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    words: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, default=lambda: datetime.now(timezone.utc)
    )


class SessionState(Base):
    """Persisted cross-invocation state for a named session.

    Columns
    -------
    session_id:
        Primary key; matches ``TranscriptionSegment.session_id``.
    processed_files:
        JSON list of audio paths already ingested (idempotency guard).
    speaker_embeddings:
        JSON dict mapping display labels to
        ``{"embedding": [...], "total_duration": float}``.
    time_cursor:
        Accumulated audio duration in seconds across all processed files.
    updated_at:
        UTC timestamp of the last update.
    """

    __tablename__ = "session_state"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    processed_files: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    speaker_embeddings: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    time_cursor: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, default=lambda: datetime.now(timezone.utc)
    )


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


# ──────────────────────────────────────────────────────────────────────────────
# Session-state helpers
# ──────────────────────────────────────────────────────────────────────────────


def load_session_state(
    session_id: str, engine
) -> Tuple[Dict[str, Any], float, List[str], int]:
    """Load prior state for a named session.

    Returns:
        ``(prior_speaker_embeddings, time_cursor, processed_files, segment_count)``.
        Returns safe defaults ``({}, 0.0, [], 0)`` when the session does not exist.
    """
    from sqlalchemy import select
    from sqlalchemy import func as sqlfunc
    with Session(engine) as db:
        row = db.get(SessionState, session_id)
        count = db.scalar(
            select(sqlfunc.count()).where(
                TranscriptionSegment.session_id == session_id
            )
        ) or 0
    if row is None:
        return {}, 0.0, [], int(count)
    return (
        row.speaker_embeddings or {},
        float(row.time_cursor or 0.0),
        list(row.processed_files or []),
        int(count),
    )


def save_session_state(
    session_id: str,
    speaker_embeddings: Dict[str, Any],
    time_cursor: float,
    processed_files: List[str],
    engine,
) -> None:
    """Upsert ``SessionState`` for *session_id*."""
    row = SessionState(
        session_id=session_id,
        speaker_embeddings=speaker_embeddings,
        time_cursor=time_cursor,
        processed_files=processed_files,
        updated_at=datetime.now(timezone.utc),
    )
    with get_session(engine) as db:
        db.merge(row)


def save_transcription_segments(
    segments: List[Dict[str, Any]],
    session_id: str,
    engine,
    start_index: int = 0,
) -> int:
    """Upsert segment dicts into ``transcription_segments``.

    Reads ``original_label`` and ``source_file`` from each segment dict
    (populated by the diarization engine).  For standalone ``transcribe``
    segments both fields may be absent and are stored as ``NULL`` / ``""``.

    Returns the number of rows upserted.
    """
    rows = []
    for i, seg in enumerate(segments):
        global_idx = start_index + i
        audio_file = seg.get("source_file") or seg.get("audio_file") or ""
        text_val = seg.get("text") or seg.get("segment") or ""
        rows.append(
            TranscriptionSegment(
                id=f"{session_id}_{global_idx}",
                session_id=session_id,
                audio_file=audio_file,
                original_speaker_label=seg.get("original_label"),
                start_time=float(seg.get("start", 0.0)),
                end_time=float(seg.get("end", 0.0)),
                text=text_val,
                words=seg.get("words") or seg.get("word_timestamps"),
                segment_index=global_idx,
                created_at=datetime.now(timezone.utc),
            )
        )
    if not rows:
        return 0
    with get_session(engine) as db:
        for row in rows:
            db.merge(row)
    return len(rows)
