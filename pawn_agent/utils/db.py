"""Minimal inline ORM models and DB session factory (no pawn_diarize dependency)."""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Float, Integer, String, Text, create_engine
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


class _Base(DeclarativeBase):
    pass


class TranscriptionSegment(_Base):
    __tablename__ = "transcription_segments"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    audio_file: Mapped[str] = mapped_column(String, nullable=False)
    original_speaker_label: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    segment_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class SpeakerName(_Base):
    __tablename__ = "speaker_names"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    audio_file: Mapped[str] = mapped_column(String, nullable=False)
    local_speaker_label: Mapped[str] = mapped_column(String, nullable=False)
    speaker_name: Mapped[str] = mapped_column(String, nullable=False)


class SessionAnalysis(_Base):
    __tablename__ = "session_analysis"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String, nullable=False)
    model: Mapped[str] = mapped_column(String, nullable=False)
    title: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    summary: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    key_topics: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    speaker_highlights: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sentiment: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    sentiment_tags: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    tags: Mapped[Optional[list]] = mapped_column(JSONB, nullable=True)
    analyzed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


def make_db_session(dsn: str) -> Session:
    engine = create_engine(dsn)
    SessionLocal = sessionmaker(bind=engine)
    return SessionLocal()
