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
import uuid
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


class SessionAnalysis(Base):
    """Persisted analysis result for a transcription session or file.

    Columns
    -------
    id:
        UUID4 primary key generated at insert time.
    session_id:
        Matches ``TranscriptionSegment.session_id`` when the analysis was
        produced from a named session.  ``NULL`` for file-based analyses.
    source:
        Human-readable label: ``"session:<id>"`` or the input file path.
    model:
        Copilot model used to produce the analysis (e.g. ``"gpt-4o"``).
    title:
        Short descriptive title generated by the model (5–10 words).
    summary:
        3–5 sentence paragraph summarising the conversation.
    key_topics:
        Bullet list of important topics and keywords.
    speaker_highlights:
        Per-speaker contribution notes.
    sentiment:
        Overall tone of the conversation (free-text prose).
    sentiment_tags:
        Up to 3 short sentiment labels for grouping/filtering
        (e.g. ``["collaborative", "formal"]``).  Stored as a JSONB array
        with a GIN index to support ``@>`` containment queries.
    tags:
        5–10 short topic/entity/tone tags for search and grouping.
        Stored as a JSONB array with a GIN index.
    analyzed_at:
        UTC timestamp of when the analysis was performed.
    """

    __tablename__ = "session_analysis"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    source: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    key_topics: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    speaker_highlights: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sentiment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sentiment_tags: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    tags: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    analyzed_at: Mapped[datetime] = mapped_column(
        DateTime, nullable=False, default=lambda: datetime.now(timezone.utc)
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
                # Prefer the resolved display name (seg["speaker"]) over the raw
                # pyannote label (seg["original_label"]) so that speaker names
                # matched from the embedding DB are persisted and visible when the
                # transcript is later loaded via _load_transcript_from_db().
                original_speaker_label=seg.get("speaker") or seg.get("original_label"),
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


def save_session_analysis(
    session_id: Optional[str],
    source: str,
    model: str,
    title: Optional[str],
    summary: Optional[str],
    key_topics: Optional[str],
    speaker_highlights: Optional[str],
    sentiment: Optional[str],
    sentiment_tags: Optional[List[str]],
    tags: Optional[List[str]],
    engine,
) -> str:
    """Insert a new analysis record into ``session_analysis``.

    Each call always inserts a new row (analyses are immutable audit records).

    Args:
        session_id: Matches ``TranscriptionSegment.session_id`` when the
            analysis was produced from a named session; ``None`` otherwise.
        source: Human-readable source label (``"session:<id>"`` or file path).
        model: Copilot model name used (e.g. ``"gpt-4o"``).
        title: Short descriptive title (5–10 words).
        summary: 3–5 sentence summary paragraph.
        key_topics: Bullet list of key topics / keywords.
        speaker_highlights: Per-speaker contribution notes.
        sentiment: Overall tone description (free-text prose).
        sentiment_tags: Up to 3 short sentiment labels for grouping/filtering.
        tags: 5–10 short topic/entity/tone tags.
        engine: A SQLAlchemy engine connected to the target database.

    Returns:
        The UUID string of the newly inserted row.
    """
    row_id = str(uuid.uuid4())
    row = SessionAnalysis(
        id=row_id,
        session_id=session_id,
        source=source,
        model=model,
        title=title,
        summary=summary,
        key_topics=key_topics,
        speaker_highlights=speaker_highlights,
        sentiment=sentiment,
        sentiment_tags=sentiment_tags,
        tags=tags,
        analyzed_at=datetime.now(timezone.utc),
    )
    with get_session(engine) as db:
        db.add(row)
    return row_id


def get_session_analysis(
    session_id: str,
    engine,
) -> Optional["SessionAnalysis"]:
    """Return the most recent :class:`SessionAnalysis` row for *session_id*.

    Selects the row with the latest ``analyzed_at`` timestamp so that
    repeated analyses return the freshest result.

    Args:
        session_id: Session identifier to look up.
        engine: A SQLAlchemy engine connected to the target database.

    Returns:
        The :class:`SessionAnalysis` ORM instance, or ``None`` when no
        analysis exists for the given *session_id*.
    """
    from sqlalchemy import select

    with Session(engine) as db:
        row = db.scalars(
            select(SessionAnalysis)
            .where(SessionAnalysis.session_id == session_id)
            .order_by(SessionAnalysis.analyzed_at.desc())
            .limit(1)
        ).first()
        if row is None:
            return None
        # Detach a plain copy so the caller can access attributes after the
        # session closes without triggering lazy-load errors.
        db.expunge(row)
        from sqlalchemy.orm import make_transient
        make_transient(row)
        return row
