"""Shared PostgreSQL ORM models and session factory for the pawn suite.

All packages (pawn_diarize, pawn_agent, pawn_server) share a single
``Base`` and the table definitions here.  Package-specific models
(e.g. ``Embedding`` in pawn_diarize, ``AgentRun`` in pawn_agent) are
defined in their respective modules and inherit from this ``Base``.

Shared tables
-------------
transcription_segments
    One diarized+transcribed segment per row.
speaker_names
    Human-readable names assigned to pyannote speaker labels.
session_analysis
    Structured analysis results (title, summary, topics, sentiment, tags).
graph_triples
    Knowledge-graph triples extracted from session transcripts.
siyuan_session_docs
    Mapping from diarization session_id to a stable SiYuan document id.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator, Optional

from sqlalchemy import Boolean, DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

# ──────────────────────────────────────────────────────────────────────────────
# Shared declarative base
# ──────────────────────────────────────────────────────────────────────────────


class Base(DeclarativeBase):
    """Single declarative base for the entire pawn suite."""


# ──────────────────────────────────────────────────────────────────────────────
# Shared ORM models
# ──────────────────────────────────────────────────────────────────────────────


class TranscriptionSegment(Base):
    """One diarized+transcribed segment stored for a session."""

    __tablename__ = "transcription_segments"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    audio_file: Mapped[str] = mapped_column(String, nullable=False)
    original_speaker_label: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    words: Mapped[Optional[Any]] = mapped_column(JSONB, nullable=True)
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, default=lambda: datetime.now(timezone.utc)
    )


class SpeakerName(Base):
    """Human-readable name assigned to a speaker in a specific audio file."""

    __tablename__ = "speaker_names"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    audio_file: Mapped[str] = mapped_column(String, nullable=False)
    local_speaker_label: Mapped[str] = mapped_column(String, nullable=False)
    speaker_name: Mapped[str] = mapped_column(String, nullable=False)
    labeled_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class SessionAnalysis(Base):
    """Persisted analysis result for a transcription session or file."""

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
    analyzed_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, default=lambda: datetime.now(timezone.utc)
    )


class GraphTriple(Base):
    """A knowledge-graph triple extracted from a session transcript."""

    __tablename__ = "graph_triples"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    subject: Mapped[str] = mapped_column(Text, nullable=False)
    relation: Mapped[str] = mapped_column(Text, nullable=False)
    object: Mapped[str] = mapped_column(Text, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    extracted_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, default=lambda: datetime.now(timezone.utc)
    )


class SiyuanSessionDoc(Base):
    """Stable SiYuan document mapping for a diarization session transcript.

    Invariant: never delete+recreate the SiYuan doc for an existing row —
    chunked diarize updates and daily-note block-refs depend on a fixed
    ``doc_id``.
    """

    __tablename__ = "siyuan_session_docs"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False)
    path: Mapped[str] = mapped_column(String, nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False, default="")
    daily_linked: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    updated_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, default=lambda: datetime.now(timezone.utc)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Session factory helpers
# ──────────────────────────────────────────────────────────────────────────────


_engines: dict[str, Engine] = {}
_engines_lock = threading.Lock()


def get_engine(dsn: str) -> Engine:
    """Return a process-wide cached :class:`Engine` for *dsn*.

    Creating a new engine on every call leaks connection pools (each orphaned
    engine keeps idle pooled connections until GC, which may never reclaim them
    while the process lives). Long-running ``pawn-server`` workers (scheduler,
    agent runner, queue) must reuse one engine per DSN.
    """
    with _engines_lock:
        engine = _engines.get(dsn)
        if engine is None:
            engine = create_engine(dsn, pool_pre_ping=True)
            _engines[dsn] = engine
        return engine


def dispose_engines() -> None:
    """Dispose and clear all cached engines (tests / process shutdown)."""
    with _engines_lock:
        engines = list(_engines.values())
        _engines.clear()
    for engine in engines:
        engine.dispose()


def make_db_session(dsn: str) -> Session:
    """Return a new :class:`Session` bound to the shared engine for *dsn*.

    Caller must ``close()`` the session (or use it as a context manager).
    """
    return sessionmaker(bind=get_engine(dsn))()


@contextmanager
def _get_session(dsn: str) -> Generator[Session, None, None]:
    """Context manager that yields a committed-or-rolled-back :class:`Session`."""
    with Session(get_engine(dsn)) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
