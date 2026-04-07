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
rag_sources
    Source documents registered in the RAG index.
text_chunks
    Text chunks with sentence-transformer embeddings for RAG retrieval.
"""

from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker

# Dimension for text chunk embeddings (sentence-transformers).
# Read from env so the value matches whatever was used when running the migration.
TEXT_CHUNK_DIM: int = int(os.environ.get("PAWN_EMBED_DIM", "1024"))


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


class RagSource(Base):
    """A source document registered in the RAG index."""

    __tablename__ = "rag_sources"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source_type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    external_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    display_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    extra_data: Mapped[Optional[Any]] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, default=lambda: datetime.now(timezone.utc)
    )


class TextChunk(Base):
    """A text chunk with a sentence-transformer embedding for RAG retrieval."""

    __tablename__ = "text_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    speaker_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    start_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    end_time: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    embedding: Mapped[list] = mapped_column(Vector(TEXT_CHUNK_DIM), nullable=False)
    extra_data: Mapped[Optional[Any]] = mapped_column("metadata", JSONB, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True, default=lambda: datetime.now(timezone.utc)
    )


# ──────────────────────────────────────────────────────────────────────────────
# Session factory helpers
# ──────────────────────────────────────────────────────────────────────────────


def make_db_session(dsn: str) -> Session:
    """Return a new :class:`Session` bound to a fresh engine for *dsn*."""
    engine = create_engine(dsn)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()


@contextmanager
def _get_session(dsn: str) -> Generator[Session, None, None]:
    """Context manager that yields a committed-or-rolled-back :class:`Session`."""
    engine = create_engine(dsn)
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise
