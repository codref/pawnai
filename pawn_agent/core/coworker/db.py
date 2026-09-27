"""PostgreSQL access for coworker items, threads, and audit rows."""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session, make_transient

from pawn_agent.utils.db import (
    AgentRun,
    CoworkerDecision,
    CoworkerItem,
    CoworkerSuppression,
    CoworkerThread,
    VaultNoteState,
    _get_session,
)
from pawn_core.database import get_engine


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _detach(db: Session, row: VaultNoteState) -> VaultNoteState:
    db.expunge(row)
    make_transient(row)
    return row


def item_dict(row: CoworkerItem) -> dict[str, Any]:
    return {
        "id": row.id,
        "short_id": row.short_id,
        "source_kind": row.source_kind,
        "source_ref": row.source_ref,
        "kind": row.kind,
        "text": row.text,
        "owner": row.owner,
        "due": row.due,
        "quote": row.quote,
        "thread": row.thread,
        "interrupt": bool(row.interrupt),
        "movement": bool(row.movement),
        "reason": row.reason,
        "fingerprint": row.fingerprint,
        "recurrence": int(row.recurrence or 0),
        "status": row.status,
        "snooze_until": row.snooze_until.isoformat() if row.snooze_until else None,
        "note_key": row.note_key,
        "payload": row.payload or {},
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "updated_at": row.updated_at.isoformat() if row.updated_at else None,
    }


def _unique_short_id(db: Session) -> str:
    for _ in range(8):
        candidate = uuid.uuid4().hex[:8]
        exists = db.scalar(select(CoworkerItem.id).where(CoworkerItem.short_id == candidate))
        if exists is None:
            return candidate
    return uuid.uuid4().hex[:12]


def triaged_fingerprints(dsn: str, source_kind: str, source_ref: str) -> set[str]:
    with Session(get_engine(dsn)) as db:
        rows = db.scalars(
            select(CoworkerItem.fingerprint).where(
                CoworkerItem.source_kind == source_kind,
                CoworkerItem.source_ref == source_ref,
                CoworkerItem.status != "new",
            )
        ).all()
    return set(rows)


def suppressed_fingerprints(dsn: str) -> set[str]:
    with Session(get_engine(dsn)) as db:
        return set(db.scalars(select(CoworkerSuppression.fingerprint)).all())


def count_notified_since(dsn: str, since: datetime) -> int:
    with Session(get_engine(dsn)) as db:
        value = db.scalar(
            select(func.count())
            .select_from(CoworkerItem)
            .where(CoworkerItem.status == "notified", CoworkerItem.updated_at >= since)
        )
    return int(value or 0)


def delete_new_for_source(dsn: str, source_kind: str, source_ref: str) -> None:
    with _get_session(dsn) as db:
        rows = db.scalars(
            select(CoworkerItem).where(
                CoworkerItem.source_kind == source_kind,
                CoworkerItem.source_ref == source_ref,
                CoworkerItem.status == "new",
            )
        ).all()
        for row in rows:
            db.delete(row)


def insert_item(dsn: str, **fields: Any) -> dict[str, Any]:
    now = _now()
    with _get_session(dsn) as db:
        row = CoworkerItem(
            id=fields.get("id") or str(uuid.uuid4()),
            short_id=fields.get("short_id") or _unique_short_id(db),
            source_kind=fields["source_kind"],
            source_ref=fields["source_ref"],
            kind=fields.get("kind") or "open_question",
            text=fields.get("text") or "",
            owner=fields.get("owner"),
            due=fields.get("due"),
            quote=fields.get("quote"),
            thread=fields.get("thread"),
            interrupt=bool(fields.get("interrupt")),
            movement=bool(fields.get("movement")),
            reason=fields.get("reason"),
            fingerprint=fields["fingerprint"],
            recurrence=int(fields.get("recurrence") or 0),
            status=fields.get("status") or "new",
            snooze_until=fields.get("snooze_until"),
            note_key=fields.get("note_key"),
            payload=fields.get("payload"),
            created_at=now,
            updated_at=now,
        )
        db.add(row)
        db.flush()
        return item_dict(row)


def get_item(dsn: str, item_id: str) -> Optional[dict[str, Any]]:
    with Session(get_engine(dsn)) as db:
        row = db.get(CoworkerItem, item_id)
        if row is None:
            row = db.scalar(select(CoworkerItem).where(CoworkerItem.short_id == item_id))
        if row is None:
            return None
        return item_dict(row)


def list_items(
    dsn: str,
    *,
    status: Optional[str] = None,
    statuses: Optional[list[str]] = None,
    limit: int = 100,
) -> list[dict[str, Any]]:
    with Session(get_engine(dsn)) as db:
        query = select(CoworkerItem).order_by(CoworkerItem.created_at.desc())
        if status:
            query = query.where(CoworkerItem.status == status)
        if statuses:
            query = query.where(CoworkerItem.status.in_(statuses))
        rows = db.scalars(query.limit(max(1, limit))).all()
        return [item_dict(row) for row in rows]


def update_item(dsn: str, item_id: str, **fields: Any) -> Optional[dict[str, Any]]:
    with _get_session(dsn) as db:
        row = db.get(CoworkerItem, item_id)
        if row is None:
            row = db.scalar(select(CoworkerItem).where(CoworkerItem.short_id == item_id))
        if row is None:
            return None
        for key, value in fields.items():
            if hasattr(row, key):
                setattr(row, key, value)
        row.updated_at = _now()
        db.flush()
        return item_dict(row)


def add_suppression(dsn: str, fingerprint_value: str) -> None:
    with _get_session(dsn) as db:
        if db.get(CoworkerSuppression, fingerprint_value) is None:
            db.add(CoworkerSuppression(fingerprint=fingerprint_value, created_at=_now()))


def unsnooze_due(dsn: str, now: Optional[datetime] = None) -> int:
    moment = now or _now()
    with _get_session(dsn) as db:
        rows = db.scalars(
            select(CoworkerItem).where(
                CoworkerItem.status == "snoozed",
                CoworkerItem.snooze_until.is_not(None),
                CoworkerItem.snooze_until <= moment,
            )
        ).all()
        for row in rows:
            row.status = "new"
            row.snooze_until = None
            row.updated_at = moment
        return len(rows)


def list_open_items(dsn: str) -> list[dict[str, Any]]:
    return list_items(
        dsn,
        statuses=["new", "notified", "snoozed", "task"],
        limit=500,
    )


def upsert_thread(
    dsn: str,
    *,
    slug: str,
    name: str,
    status: str,
    movement: bool = False,
    mention: bool = False,
    open_items: Optional[int] = None,
    now: Optional[datetime] = None,
) -> None:
    moment = now or _now()
    with _get_session(dsn) as db:
        row = db.get(CoworkerThread, slug)
        if row is None:
            row = CoworkerThread(slug=slug, name=name, status=status, created_at=moment)
            db.add(row)
        row.name = name
        row.status = status
        if movement:
            row.last_movement_at = moment
        if mention:
            row.last_mention_at = moment
        if open_items is not None:
            row.open_items = open_items


def list_threads(dsn: str) -> list[CoworkerThread]:
    with Session(get_engine(dsn)) as db:
        rows = list(db.scalars(select(CoworkerThread)).all())
        for row in rows:
            db.expunge(row)
            make_transient(row)
        return rows


def record_decision(
    dsn: str,
    *,
    event_kind: str,
    policy_decision: str,
    event_id: Optional[str] = None,
    proposed_action: Optional[str] = None,
    outcome: Optional[str] = None,
    agent_run_id: Optional[str] = None,
    error: Optional[str] = None,
) -> str:
    row_id = str(uuid.uuid4())
    with _get_session(dsn) as db:
        db.add(
            CoworkerDecision(
                id=row_id,
                event_id=event_id,
                event_kind=event_kind,
                proposed_action=proposed_action,
                policy_decision=policy_decision,
                outcome=outcome,
                agent_run_id=agent_run_id,
                error=error,
                created_at=_now(),
            )
        )
    return row_id


def get_note_state(dsn: str, key: str) -> Optional[VaultNoteState]:
    with Session(get_engine(dsn)) as db:
        row = db.get(VaultNoteState, key)
        if row is None:
            return None
        return _detach(db, row)


def upsert_note_state(dsn: str, key: str, **fields: Any) -> None:
    with _get_session(dsn) as db:
        row = db.get(VaultNoteState, key)
        if row is None:
            row = VaultNoteState(key=key)
            db.add(row)
        for name, value in fields.items():
            if hasattr(row, name):
                setattr(row, name, value)


def count_runs_for_event(dsn: str, event_id: str) -> int:
    with Session(get_engine(dsn)) as db:
        value = db.scalar(
            select(func.count()).select_from(AgentRun).where(AgentRun.event_id == event_id)
        )
    return int(value or 0)


def count_self_runs_since(dsn: str, since: datetime) -> int:
    with Session(get_engine(dsn)) as db:
        value = db.scalar(
            select(func.count())
            .select_from(AgentRun)
            .where(AgentRun.parent_run_id.is_not(None), AgentRun.created_at >= since)
        )
    return int(value or 0)


def recent_duplicate_run(dsn: str, prompt: str, session_id: str, *, hours: int = 24) -> bool:
    since = _now() - timedelta(hours=hours)
    normalized = " ".join((prompt or "").split()).lower()
    with Session(get_engine(dsn)) as db:
        rows = db.scalars(
            select(AgentRun.prompt).where(
                AgentRun.session_id == session_id,
                AgentRun.created_at >= since,
                AgentRun.parent_run_id.is_not(None),
            )
        ).all()
    for existing in rows:
        if " ".join((existing or "").split()).lower() == normalized:
            return True
    return False
