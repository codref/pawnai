"""Database manager for PawnAI.

Provides a unified interface for interacting with SQLite or PostgreSQL databases.
Handles connection management, session lifecycle, and CRUD operations.
"""

from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from pathlib import Path
import logging

from sqlalchemy import create_engine, event, Engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import QueuePool, StaticPool

from pawnai.core.models import (
    Base,
    Conversation,
    AudioFile,
    Speaker,
    SpeakerSegment,
    Transcript,
    Session as SessionModel,
    SpeakerName,
)

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Manages database connections and operations.
    
    Supports both SQLite (default, no setup) and PostgreSQL (production).
    
    Examples:
        >>> db = DatabaseManager("sqlite:///pawnai.db")
        >>> with db.get_session() as session:
        ...     audio = db.create_audio_file(session, "/path/to/audio.wav")
        ...     session.commit()
    """
    
    def __init__(self, database_url: str = "sqlite:///pawnai.db"):
        """Initialize database manager.
        
        Args:
            database_url: SQLAlchemy connection string.
                SQLite: "sqlite:///pawnai.db" or "sqlite:////absolute/path/db.sqlite"
                PostgreSQL: "postgresql://user:pass@host:port/dbname"
        """
        self.database_url = database_url
        self.is_sqlite = database_url.startswith("sqlite")
        self.is_postgresql = database_url.startswith("postgresql")
        
        # Create engine with appropriate settings
        if self.is_sqlite:
            self.engine = self._create_sqlite_engine(database_url)
        elif self.is_postgresql:
            self.engine = self._create_postgresql_engine(database_url)
        else:
            raise ValueError(f"Unsupported database type in URL: {database_url}")
        
        # Create session factory
        self.SessionLocal = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self.engine
        )
        
        logger.info(f"Database manager initialized: {self._get_db_type()}")
    
    def _create_sqlite_engine(self, database_url: str) -> Engine:
        """Create SQLite engine with optimized settings."""
        # Extract database path and ensure parent directory exists
        if database_url.startswith("sqlite:///"):
            db_path = database_url[10:]  # Remove "sqlite:///"
            if not db_path.startswith("/"):  # Relative path
                Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        
        engine = create_engine(
            database_url,
            connect_args={"check_same_thread": False},  # Allow multi-thread access
            poolclass=StaticPool,  # Single connection pool for SQLite
        )
        
        # Enable foreign keys and WAL mode for better concurrency
        @event.listens_for(engine, "connect")
        def set_sqlite_pragma(dbapi_conn, connection_record):
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.close()
        
        return engine
    
    def _create_postgresql_engine(self, database_url: str) -> Engine:
        """Create PostgreSQL engine with connection pooling."""
        return create_engine(
            database_url,
            poolclass=QueuePool,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,  # Verify connections before using
        )
    
    def _get_db_type(self) -> str:
        """Get human-readable database type."""
        if self.is_sqlite:
            return "SQLite"
        elif self.is_postgresql:
            return "PostgreSQL"
        return "Unknown"
    
    def init_db(self) -> None:
        """Create all database tables.
        
        Safe to call multiple times - only creates missing tables.
        """
        Base.metadata.create_all(bind=self.engine)
        logger.info(f"Database tables initialized ({self._get_db_type()})")
    
    def drop_all(self) -> None:
        """Drop all database tables.
        
        Warning: This will delete all data!
        """
        Base.metadata.drop_all(bind=self.engine)
        logger.warning(f"All database tables dropped ({self._get_db_type()})")
    
    @contextmanager
    def get_session(self):
        """Get a database session as a context manager.
        
        Automatically commits on success and rolls back on error.
        
        Yields:
            Session: SQLAlchemy session
            
        Example:
            >>> with db.get_session() as session:
            ...     audio = db.create_audio_file(session, "audio.wav")
            ...     session.commit()
        """
        session = self.SessionLocal()
        try:
            yield session
        except Exception:
            session.rollback()
            raise
        finally:
            session.close()
    
    # Conversation operations
    
    def create_conversation(
        self,
        session: Session,
        name: str,
        description: Optional[str] = None
    ) -> Conversation:
        """Create a new conversation."""
        conversation = Conversation(name=name, description=description)
        session.add(conversation)
        session.flush()  # Get ID without committing
        return conversation
    
    def get_conversation_by_name(
        self,
        session: Session,
        name: str
    ) -> Optional[Conversation]:
        """Get conversation by name."""
        return session.query(Conversation).filter_by(name=name).first()
    
    def get_or_create_conversation(
        self,
        session: Session,
        name: str,
        description: Optional[str] = None
    ) -> Conversation:
        """Get existing conversation or create new one."""
        conv = self.get_conversation_by_name(session, name)
        if conv is None:
            conv = self.create_conversation(session, name, description)
        return conv
    
    def list_conversations(self, session: Session) -> List[Conversation]:
        """List all conversations."""
        return session.query(Conversation).order_by(Conversation.created_at.desc()).all()
    
    # AudioFile operations
    
    def create_audio_file(
        self,
        session: Session,
        file_path: str,
        file_name: Optional[str] = None,
        duration_seconds: Optional[float] = None,
        sample_rate: Optional[int] = None,
        channels: Optional[int] = None,
        file_size_bytes: Optional[int] = None,
        conversation_id: Optional[int] = None,
    ) -> AudioFile:
        """Create a new audio file record."""
        if file_name is None:
            file_name = Path(file_path).name
        
        audio = AudioFile(
            file_path=file_path,
            file_name=file_name,
            duration_seconds=duration_seconds,
            sample_rate=sample_rate,
            channels=channels,
            file_size_bytes=file_size_bytes,
            conversation_id=conversation_id,
        )
        session.add(audio)
        session.flush()
        return audio
    
    def get_audio_file_by_path(
        self,
        session: Session,
        file_path: str
    ) -> Optional[AudioFile]:
        """Get audio file by path."""
        return session.query(AudioFile).filter_by(file_path=file_path).first()
    
    def get_or_create_audio_file(
        self,
        session: Session,
        file_path: str,
        **kwargs
    ) -> AudioFile:
        """Get existing audio file or create new one."""
        audio = self.get_audio_file_by_path(session, file_path)
        if audio is None:
            audio = self.create_audio_file(session, file_path, **kwargs)
        return audio
    
    # Speaker operations
    
    def create_speaker(
        self,
        session: Session,
        first_seen_audio_id: Optional[int] = None
    ) -> Speaker:
        """Create a new speaker."""
        speaker = Speaker(first_seen_audio_id=first_seen_audio_id)
        session.add(speaker)
        session.flush()
        return speaker
    
    def get_speaker(self, session: Session, speaker_id: int) -> Optional[Speaker]:
        """Get speaker by ID."""
        return session.query(Speaker).filter_by(id=speaker_id).first()
    
    def list_speakers(self, session: Session) -> List[Speaker]:
        """List all speakers."""
        return session.query(Speaker).order_by(Speaker.id).all()
    
    # SpeakerSegment operations
    
    def create_speaker_segment(
        self,
        session: Session,
        audio_file_id: int,
        speaker_id: int,
        start_time: float,
        end_time: float,
        local_speaker_label: Optional[str] = None,
        transcript_text: Optional[str] = None,
        confidence: Optional[float] = None,
    ) -> SpeakerSegment:
        """Create a new speaker segment."""
        segment = SpeakerSegment(
            audio_file_id=audio_file_id,
            speaker_id=speaker_id,
            start_time=start_time,
            end_time=end_time,
            local_speaker_label=local_speaker_label,
            transcript_text=transcript_text,
            confidence=confidence,
        )
        session.add(segment)
        session.flush()
        return segment
    
    def get_segments_by_audio(
        self,
        session: Session,
        audio_file_id: int
    ) -> List[SpeakerSegment]:
        """Get all segments for an audio file."""
        return (
            session.query(SpeakerSegment)
            .filter_by(audio_file_id=audio_file_id)
            .order_by(SpeakerSegment.start_time)
            .all()
        )
    
    def get_segments_by_speaker(
        self,
        session: Session,
        speaker_id: int
    ) -> List[SpeakerSegment]:
        """Get all segments for a speaker."""
        return (
            session.query(SpeakerSegment)
            .filter_by(speaker_id=speaker_id)
            .order_by(SpeakerSegment.audio_file_id, SpeakerSegment.start_time)
            .all()
        )
    
    # Transcript operations
    
    def create_transcript(
        self,
        session: Session,
        audio_file_id: int,
        text: str,
        word_timestamps: Optional[Dict] = None,
        segment_timestamps: Optional[Dict] = None,
        char_timestamps: Optional[Dict] = None,
        model_name: Optional[str] = None,
    ) -> Transcript:
        """Create a new transcript."""
        transcript = Transcript(
            audio_file_id=audio_file_id,
            text=text,
            word_timestamps=word_timestamps,
            segment_timestamps=segment_timestamps,
            char_timestamps=char_timestamps,
            model_name=model_name,
        )
        session.add(transcript)
        session.flush()
        return transcript
    
    def get_transcripts_by_audio(
        self,
        session: Session,
        audio_file_id: int
    ) -> List[Transcript]:
        """Get all transcripts for an audio file."""
        return (
            session.query(Transcript)
            .filter_by(audio_file_id=audio_file_id)
            .order_by(Transcript.created_at)
            .all()
        )
    
    # Session operations
    
    def create_session(
        self,
        session: Session,
        name: str,
        conversation_id: Optional[int] = None,
        time_cursor: float = 0.0,
        state_data: Optional[Dict] = None,
    ) -> SessionModel:
        """Create a new processing session."""
        sess = SessionModel(
            name=name,
            conversation_id=conversation_id,
            time_cursor=time_cursor,
            state_data=state_data or {},
            processed_file_ids=[],
        )
        session.add(sess)
        session.flush()
        return sess
    
    def get_session_by_name(
        self,
        session: Session,
        name: str
    ) -> Optional[SessionModel]:
        """Get session by name."""
        return session.query(SessionModel).filter_by(name=name).first()
    
    def get_or_create_session(
        self,
        session: Session,
        name: str,
        **kwargs
    ) -> SessionModel:
        """Get existing session or create new one."""
        sess = self.get_session_by_name(session, name)
        if sess is None:
            sess = self.create_session(session, name, **kwargs)
        return sess
    
    def update_session(
        self,
        session: Session,
        session_id: int,
        time_cursor: Optional[float] = None,
        state_data: Optional[Dict] = None,
        processed_file_ids: Optional[List[int]] = None,
    ) -> SessionModel:
        """Update session state."""
        sess = session.query(SessionModel).filter_by(id=session_id).first()
        if sess is None:
            raise ValueError(f"Session {session_id} not found")
        
        if time_cursor is not None:
            sess.time_cursor = time_cursor
        if state_data is not None:
            sess.state_data = state_data
        if processed_file_ids is not None:
            sess.processed_file_ids = processed_file_ids
        
        session.flush()
        return sess
    
    def list_sessions(self, session: Session) -> List[SessionModel]:
        """List all sessions."""
        return session.query(SessionModel).order_by(SessionModel.updated_at.desc()).all()
    
    # SpeakerName operations
    
    def create_speaker_name(
        self,
        session: Session,
        speaker_id: int,
        name: str,
        context_audio_id: Optional[int] = None,
    ) -> SpeakerName:
        """Create or update speaker name."""
        # Check if name already exists for this speaker
        existing = (
            session.query(SpeakerName)
            .filter_by(speaker_id=speaker_id)
            .first()
        )
        
        if existing:
            existing.name = name
            existing.context_audio_id = context_audio_id
            session.flush()
            return existing
        
        speaker_name = SpeakerName(
            speaker_id=speaker_id,
            name=name,
            context_audio_id=context_audio_id,
        )
        session.add(speaker_name)
        session.flush()
        return speaker_name
    
    def get_speaker_name(
        self,
        session: Session,
        speaker_id: int
    ) -> Optional[SpeakerName]:
        """Get speaker name by speaker ID."""
        return session.query(SpeakerName).filter_by(speaker_id=speaker_id).first()
    
    def list_speaker_names(self, session: Session) -> List[SpeakerName]:
        """List all speaker names."""
        return session.query(SpeakerName).order_by(SpeakerName.speaker_id).all()
    
    def delete_speaker_name(
        self,
        session: Session,
        speaker_id: int
    ) -> bool:
        """Delete speaker name."""
        speaker_name = self.get_speaker_name(session, speaker_id)
        if speaker_name:
            session.delete(speaker_name)
            session.flush()
            return True
        return False
