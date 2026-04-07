"""PostgreSQL database layer using SQLAlchemy ORM + pgvector.

This module defines the pawn-diarize–specific ORM models, engine factory, and
session helper.  Shared models (TranscriptionSegment, SpeakerName, SessionAnalysis,
GraphTriple, RagSource, TextChunk) are imported from pawn_core.database.

Tables owned here
-----------------
embeddings
    Stores per-segment speaker embeddings extracted by pyannote.audio.
    The ``embedding`` column is a pgvector ``vector(512)`` for cosine-similarity
    searches.

session_state
    Cross-invocation state for named sessions (processed files, time cursor).
"""

from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Dict, Generator, List, Optional, Tuple

from sqlalchemy import DateTime, Float, String, Text, create_engine, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, Session, mapped_column

from pgvector.sqlalchemy import Vector

from pawn_core.database import (  # noqa: F401
    Base,
    GraphTriple,
    RagSource,
    SessionAnalysis,
    SpeakerName,
    TEXT_CHUNK_DIM,
    TextChunk,
    TranscriptionSegment,
    _get_session,
    make_db_session,
)


# Dimension produced by pyannote/embedding model.
EMBEDDING_DIM = 512


# ──────────────────────────────────────────────────────────────────────────────
# Diarize-specific ORM Models
# ──────────────────────────────────────────────────────────────────────────────


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
             ``"postgresql+psycopg://postgres:postgres@localhost:5432/pawn_diarize"``

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


def save_graph_triples(
    session_id: str,
    triples: List[Tuple[str, str, str]],
    model: str,
    engine,
) -> int:
    """Bulk-insert knowledge-graph triples for *session_id*.

    Deletes any previously stored triples for the session before inserting
    so that re-running the tool gives a fresh result rather than duplicates.

    Args:
        session_id: Session identifier the triples were extracted from.
        triples: List of ``(subject, relation, object)`` tuples.
        model: Copilot model name used (e.g. ``"gpt-4o"``).
        engine: A SQLAlchemy engine connected to the target database.

    Returns:
        Number of triples inserted.
    """
    from sqlalchemy import delete

    now = datetime.now(timezone.utc)
    rows = [
        GraphTriple(
            id=str(uuid.uuid4()),
            session_id=session_id,
            subject=s,
            relation=r,
            object=o,
            model=model,
            extracted_at=now,
        )
        for s, r, o in triples
    ]
    with get_session(engine) as db:
        db.execute(delete(GraphTriple).where(GraphTriple.session_id == session_id))
        db.add_all(rows)
    return len(rows)


def get_graph_triples(
    session_id: str,
    engine,
) -> List["GraphTriple"]:
    """Return all :class:`GraphTriple` rows for *session_id*.

    Args:
        session_id: Session identifier to look up.
        engine: A SQLAlchemy engine connected to the target database.

    Returns:
        List of :class:`GraphTriple` instances (may be empty).
    """
    from sqlalchemy import select
    from sqlalchemy.orm import make_transient

    with Session(engine) as db:
        rows = list(
            db.scalars(
                select(GraphTriple).where(GraphTriple.session_id == session_id)
            ).all()
        )
        for row in rows:
            db.expunge(row)
            make_transient(row)
    return rows
