"""ORM models and DB session factory for pawn-agent."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, List, Optional, Tuple

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, Session, mapped_column

from pawn_core.database import Base as _Base  # noqa: F401
from pawn_core.database import (
    GraphTriple,
    SessionAnalysis,
    SpeakerName,
    TranscriptionSegment,
    _get_session,
    get_engine,
    make_db_session,
)
from pawn_core.vault import normalize_vault_key


class AgentRun(_Base):
    """Persists every agent execution for history / auditability."""

    __tablename__ = "agent_runs"
    id: Mapped[str] = mapped_column(String, primary_key=True)
    message_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    source: Mapped[str] = mapped_column(String, nullable=False, default="queue", index=True)
    schedule_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    scheduled_fire_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    command: Mapped[str] = mapped_column(String, nullable=False)
    prompt: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    model: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending")
    response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    parent_run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    depth: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    event_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)


class AgentSchedule(_Base):
    """A durable schedule for future pawn-agent turns."""

    __tablename__ = "agent_schedules"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    prompt: Mapped[str] = mapped_column(Text, nullable=False)
    session_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="active", index=True)
    schedule_kind: Mapped[str] = mapped_column(String, nullable=False)
    timezone: Mapped[str] = mapped_column(String, nullable=False)
    run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    interval_seconds: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    cron_expression: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    next_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)
    last_run_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_by: Mapped[str] = mapped_column(String, nullable=False, default="user")
    created_from_proposal_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    metadata_json: Mapped[Optional[Any]] = mapped_column("metadata", JSON, nullable=True)
    output_note: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class AgentScheduleProposal(_Base):
    """A model- or user-created proposal for a schedule mutation."""

    __tablename__ = "agent_schedule_proposals"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    action: Mapped[str] = mapped_column(String, nullable=False, index=True)
    schedule_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("agent_schedules.id"), nullable=True, index=True
    )
    proposed_payload: Mapped[dict[str, Any]] = mapped_column(JSON, nullable=False)
    rationale: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="proposed", index=True)
    proposed_by_run_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("agent_runs.id"), nullable=True
    )
    proposed_by_session_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    reviewed_by: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    reviewed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class AgentScheduleFire(_Base):
    """One due execution attempt for an agent schedule."""

    __tablename__ = "agent_schedule_fires"
    __table_args__ = (
        UniqueConstraint("schedule_id", "scheduled_for", name="uq_agent_schedule_fire_once"),
    )

    id: Mapped[str] = mapped_column(String, primary_key=True)
    schedule_id: Mapped[str] = mapped_column(
        String, ForeignKey("agent_schedules.id"), nullable=False, index=True
    )
    scheduled_for: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="claimed", index=True)
    agent_run_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("agent_runs.id"), nullable=True, index=True
    )
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    claimed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class ApiIpBlacklist(_Base):
    """Client IPs blocked by API brute-force / scan heuristics."""

    __tablename__ = "api_ip_blacklist"

    ip: Mapped[str] = mapped_column(String, primary_key=True)
    reason: Mapped[str] = mapped_column(String, nullable=False)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)
    hit_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class VaultTask(_Base):
    """Durable lifecycle for one vault-sourced agent task note."""

    __tablename__ = "vault_tasks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    key: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="queued", index=True)
    conversation_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    instruction_hash: Mapped[str] = mapped_column(String, nullable=False)
    etag: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    via: Mapped[str] = mapped_column(String, nullable=False, default="vault")
    agent_run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    matrix_notify_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    instruction_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    note_path: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    error_code: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    # Job kind: ask | push_note | upload (vault-watcher tasks are always ask).
    kind: Mapped[str] = mapped_column(String, nullable=False, default="ask")
    # Kind-specific request data and outcome (e.g. upload target key).
    payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    result_text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)


class CoworkerItem(_Base):
    """One extracted decision, commitment, question, or follow-up proposal."""

    __tablename__ = "coworker_items"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    short_id: Mapped[str] = mapped_column(String, nullable=False, unique=True, index=True)
    source_kind: Mapped[str] = mapped_column(String, nullable=False, index=True)
    source_ref: Mapped[str] = mapped_column(String, nullable=False, index=True)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    owner: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    due: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    quote: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    thread: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    interrupt: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    movement: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    fingerprint: Mapped[str] = mapped_column(String, nullable=False, index=True)
    recurrence: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String, nullable=False, default="new", index=True)
    snooze_until: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    note_key: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class CoworkerSuppression(_Base):
    """Fingerprints the user asked Pawn to stop surfacing."""

    __tablename__ = "coworker_suppressions"

    fingerprint: Mapped[str] = mapped_column(String, primary_key=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class VaultNoteState(_Base):
    """Etag ledger so the vault scanner does not reprocess quiet notes."""

    __tablename__ = "vault_note_state"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    etag: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    content_hash: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_processed_hash: Mapped[Optional[str]] = mapped_column(String, nullable=True)


class KnowledgeChunk(_Base):
    """Shared embedding of a note, transcript, analysis, or coworker item."""

    __tablename__ = "knowledge_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source_kind: Mapped[str] = mapped_column(String, nullable=False, index=True)
    source_ref: Mapped[str] = mapped_column(String, nullable=False, index=True)
    heading: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[list] = mapped_column(Vector(1024), nullable=False)
    content_hash: Mapped[str] = mapped_column(String, nullable=False, index=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class Device(_Base):
    """A registered phone (pawn-assistant) that pushes context and runs tools."""

    __tablename__ = "devices"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    token_hash: Mapped[str] = mapped_column(String, nullable=False)
    unifiedpush_endpoint: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    enabled_categories: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    # Live tools the phone will run (location_now, status_now, sms_send, notify).
    capabilities: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_seen: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class PhoneEvent(_Base):
    """One opt-in phone event (call, sms, notification, location, status)."""

    __tablename__ = "phone_events"
    __table_args__ = (UniqueConstraint("device_id", "dedup_key", name="uq_phone_events_dedup"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    device_id: Mapped[str] = mapped_column(
        String, ForeignKey("devices.id", ondelete="CASCADE"), nullable=False, index=True
    )
    kind: Mapped[str] = mapped_column(String, nullable=False, index=True)
    ts: Mapped[datetime] = mapped_column(DateTime, nullable=False, index=True)
    dedup_key: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    text: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    received_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class DeviceCommand(_Base):
    """A tool call queued for a phone (live read or gated action)."""

    __tablename__ = "device_commands"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    device_id: Mapped[str] = mapped_column(
        String, ForeignKey("devices.id", ondelete="CASCADE"), nullable=False, index=True
    )
    tool: Mapped[str] = mapped_column(String, nullable=False)
    args: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="pending", index=True)
    result: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class CoworkerThread(_Base):
    """Movement tracking for one goals thread."""

    __tablename__ = "coworker_threads"

    slug: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="active", index=True)
    last_movement_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    last_mention_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    open_items: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class CoworkerDecision(_Base):
    """Audit row for an autonomy or notification decision."""

    __tablename__ = "coworker_decisions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    event_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    event_kind: Mapped[str] = mapped_column(String, nullable=False)
    proposed_action: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    policy_decision: Mapped[str] = mapped_column(String, nullable=False)
    outcome: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    agent_run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class EmSource(_Base):
    """One processed source for the emergence layer (session, note, mail, …)."""

    __tablename__ = "em_sources"
    __table_args__ = (UniqueConstraint("kind", "ref", name="uq_em_sources_kind_ref"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    kind: Mapped[str] = mapped_column(String, nullable=False)
    ref: Mapped[str] = mapped_column(String, nullable=False, index=True)
    occurred_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    content_hash: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    pipeline_version: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    status: Mapped[str] = mapped_column(String, nullable=False, default="done")
    stats: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    processed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class EmWindow(_Base):
    """One slice of a source with its triage scores (latest run only)."""

    __tablename__ = "em_windows"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source_id: Mapped[str] = mapped_column(
        String, ForeignKey("em_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    run_id: Mapped[str] = mapped_column(String, nullable=False)
    ordinal: Mapped[int] = mapped_column(Integer, nullable=False)
    char_start: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    char_end: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    start_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    end_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    content_hash: Mapped[str] = mapped_column(String, nullable=False, index=True)
    triage: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    triage_model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    embedding: Mapped[Optional[list]] = mapped_column(Vector(1024), nullable=True)


class EmObservation(_Base):
    """One atomic, typed, quoted claim extracted from a window."""

    __tablename__ = "em_observations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    source_id: Mapped[str] = mapped_column(
        String, ForeignKey("em_sources.id", ondelete="CASCADE"), nullable=False, index=True
    )
    window_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("em_windows.id", ondelete="SET NULL"), nullable=True
    )
    run_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    speaker_name: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    speaker_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    quote: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    quote_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    start_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    fields: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    extractor: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    embedding: Mapped[Optional[list]] = mapped_column(Vector(1024), nullable=True)
    # live | rejected | superseded
    status: Mapped[str] = mapped_column(String, nullable=False, default="live", index=True)
    reject_reason: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class EmLabel(_Base):
    """A user's verdict on an observation, item or review line (training row)."""

    __tablename__ = "em_labels"
    __table_args__ = (UniqueConstraint("target_kind", "target_id", name="uq_em_labels_target"),)

    id: Mapped[str] = mapped_column(String, primary_key=True)
    # item | observation | review | window
    target_kind: Mapped[str] = mapped_column(String, nullable=False)
    target_id: Mapped[str] = mapped_column(String, nullable=False)
    # accept | reject | edit | missed
    verdict: Mapped[str] = mapped_column(String, nullable=False, index=True)
    # inbox | matrix | note | chat | review | bulk | api
    surface: Mapped[str] = mapped_column(String, nullable=False)
    # Observation type (decision, commitment, question, …).
    type: Mapped[str] = mapped_column(String, nullable=False, index=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    quote: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    context: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    source_ref: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    start_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    observation_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    item_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    extractor: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    features: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    before: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    after: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    embedding: Mapped[Optional[list]] = mapped_column(Vector(1024), nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


def get_session_analysis(session_id: str, dsn: str) -> Optional[SessionAnalysis]:
    """Return the most recent SessionAnalysis row for *session_id*, or None."""
    from sqlalchemy import select

    with Session(get_engine(dsn)) as db:
        row = db.scalars(
            select(SessionAnalysis)
            .where(SessionAnalysis.session_id == session_id)
            .order_by(SessionAnalysis.analyzed_at.desc())
            .limit(1)
        ).first()
        if row is None:
            return None
        db.expunge(row)
        from sqlalchemy.orm import make_transient

        make_transient(row)
        return row


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
    dsn: str,
) -> str:
    """Insert a new row into ``session_analysis`` and return its UUID."""
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
    with _get_session(dsn) as db:
        db.add(row)
    return row_id


def save_graph_triples(
    session_id: str,
    triples: List[Tuple[str, str, str]],
    model: str,
    dsn: str,
) -> int:
    """Delete existing triples for *session_id* and bulk-insert new ones.

    Returns the number of triples inserted.
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
    with _get_session(dsn) as db:
        db.execute(delete(GraphTriple).where(GraphTriple.session_id == session_id))
        db.add_all(rows)
    return len(rows)


# ---------------------------------------------------------------------------
# AgentRun helpers
# ---------------------------------------------------------------------------


def create_agent_run(
    dsn: str,
    *,
    message_id: Optional[str] = None,
    source: str = "queue",
    schedule_id: Optional[str] = None,
    scheduled_fire_id: Optional[str] = None,
    command: str,
    prompt: Optional[str] = None,
    session_id: Optional[str] = None,
    model: str,
    parent_run_id: Optional[str] = None,
    depth: int = 0,
    event_id: Optional[str] = None,
) -> str:
    """Insert a new ``agent_runs`` row with status ``pending``. Returns the UUID."""
    row_id = str(uuid.uuid4())
    row = AgentRun(
        id=row_id,
        message_id=message_id,
        source=source,
        schedule_id=schedule_id,
        scheduled_fire_id=scheduled_fire_id,
        command=command,
        prompt=prompt,
        session_id=session_id,
        model=model,
        status="pending",
        created_at=datetime.now(timezone.utc),
        parent_run_id=parent_run_id,
        depth=depth,
        event_id=event_id,
    )
    with _get_session(dsn) as db:
        db.add(row)
    return row_id


def update_agent_run(
    dsn: str,
    run_id: str,
    status: str,
    *,
    response: Optional[str] = None,
    error: Optional[str] = None,
) -> None:
    """Update an ``agent_runs`` row's status and optional response/error."""
    now = datetime.now(timezone.utc)
    with _get_session(dsn) as db:
        row = db.get(AgentRun, run_id)
        if row is None:
            return
        row.status = status
        if status == "running":
            row.started_at = now
        if status in ("completed", "failed"):
            row.completed_at = now
        if response is not None:
            row.response = response
        if error is not None:
            row.error = error


# ---------------------------------------------------------------------------
# Vault task helpers
# ---------------------------------------------------------------------------


def upsert_vault_task(
    dsn: str,
    *,
    task_id: str,
    key: str,
    instruction_hash: str,
    conversation_id: str,
    instruction_text: Optional[str] = None,
    note_path: Optional[str] = None,
    etag: Optional[str] = None,
    via: str = "vault",
    status: str = "queued",
    kind: str = "ask",
    payload: Optional[dict] = None,
) -> str:
    """Insert or refresh a task for *key* + *instruction_hash*."""
    now = datetime.now(timezone.utc)
    with _get_session(dsn) as db:
        by_hash = (
            db.query(VaultTask).filter_by(key=key, instruction_hash=instruction_hash).one_or_none()
        )
        if by_hash is not None:
            return by_hash.id

        open_queued = (
            db.query(VaultTask)
            .filter_by(key=key, status="queued")
            .order_by(VaultTask.created_at.desc())
            .first()
        )
        if open_queued is not None:
            open_queued.instruction_hash = instruction_hash
            open_queued.instruction_text = instruction_text
            open_queued.note_path = note_path
            open_queued.etag = etag
            open_queued.conversation_id = conversation_id
            open_queued.updated_at = now
            return open_queued.id

        effective_id = task_id
        if db.get(VaultTask, effective_id) is not None:
            effective_id = str(uuid.uuid4())

        row = VaultTask(
            id=effective_id,
            key=key,
            instruction_hash=instruction_hash,
            status=status,
            conversation_id=conversation_id,
            instruction_text=instruction_text,
            note_path=note_path,
            etag=etag,
            via=via,
            kind=kind,
            payload=payload,
            created_at=now,
            updated_at=now,
        )
        db.add(row)
    return effective_id


def get_vault_task(dsn: str, task_id: str) -> Optional[VaultTask]:
    """Return a detached vault task row or None."""
    with Session(get_engine(dsn)) as db:
        row = db.get(VaultTask, task_id)
        if row is None:
            return None
        db.expunge(row)
        from sqlalchemy.orm import make_transient

        make_transient(row)
        return row


def get_vault_task_by_key(dsn: str, key: str) -> Optional[VaultTask]:
    """Return the vault task row for *key*, or None."""
    with Session(get_engine(dsn)) as db:
        row = db.query(VaultTask).filter_by(key=normalize_vault_key(key)).one_or_none()
        if row is None:
            return None
        db.expunge(row)
        from sqlalchemy.orm import make_transient

        make_transient(row)
        return row


def list_vault_tasks(
    dsn: str,
    *,
    status: Optional[str] = None,
    statuses: Optional[List[str]] = None,
    conversation_id: Optional[str] = None,
    newest_first: bool = False,
    limit: int = 50,
) -> List[VaultTask]:
    """List vault task rows, optionally filtered."""
    with Session(get_engine(dsn)) as db:
        q = db.query(VaultTask)
        if status:
            q = q.filter(VaultTask.status == status)
        if statuses:
            q = q.filter(VaultTask.status.in_(statuses))
        if conversation_id:
            q = q.filter(VaultTask.conversation_id == conversation_id)
        order = VaultTask.created_at.desc() if newest_first else VaultTask.created_at.asc()
        rows = q.order_by(order).limit(limit).all()
        out: List[VaultTask] = []
        from sqlalchemy.orm import make_transient

        for row in rows:
            db.expunge(row)
            make_transient(row)
            out.append(row)
        return out


def update_vault_task(
    dsn: str,
    task_id: str,
    *,
    status: Optional[str] = None,
    agent_run_id: Optional[str] = None,
    matrix_notify_id: Optional[str] = None,
    error_code: Optional[str] = None,
    indexed_at: Optional[datetime] = None,
    etag: Optional[str] = None,
    result_text: Optional[str] = None,
    payload: Optional[dict] = None,
) -> None:
    """Patch mutable fields on a vault task."""
    now = datetime.now(timezone.utc)
    with _get_session(dsn) as db:
        row = db.get(VaultTask, task_id)
        if row is None:
            return
        if status is not None:
            row.status = status
        if agent_run_id is not None:
            row.agent_run_id = agent_run_id
        if matrix_notify_id is not None:
            row.matrix_notify_id = matrix_notify_id
        if error_code is not None:
            row.error_code = error_code
        if indexed_at is not None:
            row.indexed_at = indexed_at
        if etag is not None:
            row.etag = etag
        if result_text is not None:
            row.result_text = result_text
        if payload is not None:
            row.payload = payload
        row.updated_at = now


def claim_vault_task(dsn: str, task_id: str) -> bool:
    """Atomically move ``queued`` → ``claimed``. Returns False if not claimable."""
    now = datetime.now(timezone.utc)
    with _get_session(dsn) as db:
        row = db.get(VaultTask, task_id)
        if row is None or row.status != "queued":
            return False
        row.status = "claimed"
        row.updated_at = now
        return True


def delete_vault_task(dsn: str, task_id: str) -> bool:
    """Delete one vault task row. Returns True if a row was removed."""
    with _get_session(dsn) as db:
        row = db.get(VaultTask, task_id)
        if row is None:
            return False
        db.delete(row)
        return True


# ---------------------------------------------------------------------------
# API IP blacklist helpers
# ---------------------------------------------------------------------------


def _blacklist_active(row: ApiIpBlacklist, now: datetime) -> bool:
    if row.expires_at is None:
        return True
    expires = row.expires_at
    if expires.tzinfo is None:
        expires = expires.replace(tzinfo=timezone.utc)
    return expires > now


def is_ip_blacklisted(dsn: str, ip: str) -> bool:
    """Return True if *ip* has an active blacklist row."""
    now = datetime.now(timezone.utc)
    with Session(get_engine(dsn)) as db:
        row = db.get(ApiIpBlacklist, ip)
        if row is None:
            return False
        if not _blacklist_active(row, now):
            db.delete(row)
            db.commit()
            return False
        return True


def add_ip_blacklist(
    dsn: str,
    ip: str,
    *,
    reason: str,
    ttl_seconds: Optional[int] = None,
    hit_count: int = 0,
) -> ApiIpBlacklist:
    """Insert or refresh a blacklist entry. Returns a detached row."""
    now = datetime.now(timezone.utc)
    expires = now + timedelta(seconds=ttl_seconds) if ttl_seconds else None
    with _get_session(dsn) as db:
        row = db.get(ApiIpBlacklist, ip)
        if row is None:
            row = ApiIpBlacklist(
                ip=ip,
                reason=reason,
                created_at=now,
                expires_at=expires,
                hit_count=hit_count,
            )
            db.add(row)
        else:
            row.reason = reason
            row.expires_at = expires
            if hit_count:
                row.hit_count = hit_count
        db.flush()
        db.expunge(row)
        from sqlalchemy.orm import make_transient

        make_transient(row)
        return row


def remove_ip_blacklist(dsn: str, ip: str) -> bool:
    """Delete a blacklist row. Returns True if a row was removed."""
    with _get_session(dsn) as db:
        row = db.get(ApiIpBlacklist, ip)
        if row is None:
            return False
        db.delete(row)
        return True


def list_ip_blacklist(dsn: str, *, include_expired: bool = False) -> List[ApiIpBlacklist]:
    """Return blacklist rows, optionally including expired ones."""
    now = datetime.now(timezone.utc)
    with Session(get_engine(dsn)) as db:
        rows = db.query(ApiIpBlacklist).order_by(ApiIpBlacklist.created_at.desc()).all()
        out: List[ApiIpBlacklist] = []
        from sqlalchemy.orm import make_transient

        for row in rows:
            active = _blacklist_active(row, now)
            if not active and not include_expired:
                continue
            db.expunge(row)
            make_transient(row)
            out.append(row)
        return out


def clear_ip_blacklist(dsn: str) -> int:
    """Delete every blacklist row. Returns the number removed."""
    with _get_session(dsn) as db:
        rows = db.query(ApiIpBlacklist).all()
        count = len(rows)
        for row in rows:
            db.delete(row)
        return count
