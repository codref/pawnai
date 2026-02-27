"""SQLAlchemy database models for PawnAI.

This module defines the database schema for storing audio files, transcripts,
speaker information, and diarization results. Supports both SQLite and PostgreSQL.
"""

from datetime import datetime
from typing import Optional, List
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
    Index,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Base class for all database models."""
    pass


class Conversation(Base):
    """Conversation groups related audio files together.
    
    Examples: a meeting, podcast episode, interview, or project.
    """
    __tablename__ = "conversations"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    
    # Relationships
    audio_files: Mapped[List["AudioFile"]] = relationship(
        "AudioFile", back_populates="conversation", cascade="all, delete-orphan"
    )
    sessions: Mapped[List["Session"]] = relationship(
        "Session", back_populates="conversation", cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Conversation(id={self.id}, name='{self.name}')>"


class AudioFile(Base):
    """Audio file record with metadata."""
    __tablename__ = "audio_files"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    file_path: Mapped[str] = mapped_column(String(512), nullable=False, unique=True)
    file_name: Mapped[str] = mapped_column(String(255), nullable=False)
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sample_rate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    channels: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    file_size_bytes: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    conversation_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    
    # Relationships
    conversation: Mapped[Optional["Conversation"]] = relationship(
        "Conversation", back_populates="audio_files"
    )
    segments: Mapped[List["SpeakerSegment"]] = relationship(
        "SpeakerSegment", back_populates="audio_file", cascade="all, delete-orphan"
    )
    transcripts: Mapped[List["Transcript"]] = relationship(
        "Transcript", back_populates="audio_file", cascade="all, delete-orphan"
    )
    
    __table_args__ = (
        Index("idx_audio_file_path", "file_path"),
        Index("idx_audio_conversation", "conversation_id"),
    )
    
    def __repr__(self) -> str:
        return f"<AudioFile(id={self.id}, name='{self.file_name}')>"


class Speaker(Base):
    """Global speaker record with unique ID.
    
    Speakers are identified across audio files through embedding similarity.
    """
    __tablename__ = "speakers"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    # First audio file where speaker was detected
    first_seen_audio_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("audio_files.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    segments: Mapped[List["SpeakerSegment"]] = relationship(
        "SpeakerSegment", back_populates="speaker"
    )
    names: Mapped[List["SpeakerName"]] = relationship(
        "SpeakerName", back_populates="speaker", cascade="all, delete-orphan"
    )
    
    def __repr__(self) -> str:
        return f"<Speaker(id={self.id})>"


class SpeakerSegment(Base):
    """Speaker diarization segment - who spoke when.
    
    Records a continuous speech segment by a specific speaker.
    """
    __tablename__ = "speaker_segments"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio_file_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("audio_files.id", ondelete="CASCADE"), nullable=False
    )
    speaker_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("speakers.id", ondelete="CASCADE"), nullable=False
    )
    start_time: Mapped[float] = mapped_column(Float, nullable=False)
    end_time: Mapped[float] = mapped_column(Float, nullable=False)
    # Local label from diarization pipeline (e.g., "SPEAKER_00")
    local_speaker_label: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    # Transcribed text for this segment (if combined transcribe-diarize)
    transcript_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    # Confidence score from diarization model
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    audio_file: Mapped["AudioFile"] = relationship("AudioFile", back_populates="segments")
    speaker: Mapped["Speaker"] = relationship("Speaker", back_populates="segments")
    
    __table_args__ = (
        Index("idx_segment_audio_speaker", "audio_file_id", "speaker_id"),
        Index("idx_segment_times", "audio_file_id", "start_time", "end_time"),
    )
    
    def __repr__(self) -> str:
        return f"<SpeakerSegment(id={self.id}, speaker_id={self.speaker_id}, {self.start_time:.2f}-{self.end_time:.2f})>"


class Transcript(Base):
    """Full transcription result for an audio file.
    
    Stores the complete transcript with word-level timestamps.
    """
    __tablename__ = "transcripts"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    audio_file_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("audio_files.id", ondelete="CASCADE"), nullable=False
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # JSON structure: [{"word": "hello", "start": 0.5, "end": 0.8}, ...]
    word_timestamps: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # JSON structure: [{"start": 0.0, "end": 5.2, "text": "..."}, ...]
    segment_timestamps: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # JSON structure: [{"char": "h", "start": 0.5, "end": 0.52}, ...]
    char_timestamps: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # Model used for transcription
    model_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    
    # Relationships
    audio_file: Mapped["AudioFile"] = relationship("AudioFile", back_populates="transcripts")
    
    __table_args__ = (
        Index("idx_transcript_audio", "audio_file_id"),
    )
    
    def __repr__(self) -> str:
        preview = self.text[:50] + "..." if len(self.text) > 50 else self.text
        return f"<Transcript(id={self.id}, audio_id={self.audio_file_id}, text='{preview}')>"


class Session(Base):
    """Incremental processing session for multi-file workflows.
    
    Replaces JSON file-based session management with database storage.
    """
    __tablename__ = "sessions"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    conversation_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("conversations.id", ondelete="CASCADE"), nullable=True
    )
    # Cumulative time processed (seconds)
    time_cursor: Mapped[float] = mapped_column(Float, default=0.0)
    # JSON: Session state including speaker embeddings
    # {"session_speaker_embeddings": {...}, "segments": [...], "word_timestamps": [...]}
    state_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # List of audio file IDs processed in this session
    processed_file_ids: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    
    # Relationships
    conversation: Mapped[Optional["Conversation"]] = relationship(
        "Conversation", back_populates="sessions"
    )
    
    __table_args__ = (
        Index("idx_session_name", "name"),
        Index("idx_session_conversation", "conversation_id"),
    )
    
    def __repr__(self) -> str:
        return f"<Session(id={self.id}, name='{self.name}', time={self.time_cursor})>"


class SpeakerName(Base):
    """Human-readable label for a speaker.
    
    Maps speaker IDs to names (e.g., "John Doe", "Host").
    """
    __tablename__ = "speaker_names"
    
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    speaker_id: Mapped[int] = mapped_column(
        Integer, ForeignKey("speakers.id", ondelete="CASCADE"), nullable=False
    )
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    # Optional context (e.g., which audio file this name was assigned in)
    context_audio_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("audio_files.id", ondelete="SET NULL"), nullable=True
    )
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )
    
    # Relationships
    speaker: Mapped["Speaker"] = relationship("Speaker", back_populates="names")
    
    __table_args__ = (
        Index("idx_speaker_name_speaker", "speaker_id"),
    )
    
    def __repr__(self) -> str:
        return f"<SpeakerName(id={self.id}, speaker_id={self.speaker_id}, name='{self.name}')>"
