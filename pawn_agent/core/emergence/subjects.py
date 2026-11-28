"""Storage for subjects, links, subject events and helper vector math."""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

import numpy as np
from sqlalchemy import func, select

from pawn_agent.utils.db import (
    EmLink,
    EmObservation,
    EmSource,
    EmSubject,
    EmSubjectEvent,
    _get_session,
)

SUBJECT_KINDS = ("thread", "question", "decision", "commitment", "prediction", "hypothesis")
# Statuses that still take evidence.
OPEN_STATUSES = ("proposed", "active", "open", "parked", "pending")


def now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def slugify(text: str, limit: int = 60) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return (slug[:limit].rstrip("-")) or "subject"


def unit(vector: Optional[Iterable[float]]) -> Optional[np.ndarray]:
    if vector is None:
        return None
    arr = np.asarray(list(vector), dtype=np.float64)
    if arr.size == 0:
        return None
    norm = np.linalg.norm(arr)
    return arr / norm if norm > 0 else None


def cosine_distance(a: Any, b: Any) -> float:
    ua, ub = unit(a), unit(b)
    if ua is None or ub is None or ua.shape != ub.shape:
        return 2.0
    return float(1.0 - ua @ ub)


def subject_dict(row: EmSubject) -> dict[str, Any]:
    return {
        "id": row.id,
        "kind": row.kind,
        "title": row.title,
        "slug": row.slug,
        "origin": row.origin,
        "status": row.status,
        "state_md": row.state_md or "",
        "fields": dict(row.fields or {}),
        "momentum": row.momentum,
        "source_ref": row.source_ref,
        "parent_id": row.parent_id,
        "last_evidence_at": row.last_evidence_at,
        "state_at": row.state_at,
        "checked_at": row.checked_at,
        "review_at": row.review_at,
        "note_key": row.note_key,
        "centroid": [float(x) for x in row.centroid] if row.centroid is not None else None,
        "created_at": row.created_at,
        "updated_at": row.updated_at,
    }


def create_subject(dsn: str, **fields: Any) -> dict[str, Any]:
    stamp = now()
    with _get_session(dsn) as db:
        row = EmSubject(
            id=fields.pop("id", None) or str(uuid.uuid4()),
            slug=fields.pop("slug", None) or slugify(fields.get("title") or ""),
            created_at=stamp,
            updated_at=stamp,
            **fields,
        )
        db.add(row)
        db.flush()
        db.add(
            EmSubjectEvent(
                id=str(uuid.uuid4()),
                subject_id=row.id,
                at=stamp,
                kind="created",
                diff={"title": row.title, "status": row.status, "origin": row.origin},
            )
        )
        return subject_dict(row)


def update_subject(dsn: str, subject_id: str, **fields: Any) -> Optional[dict[str, Any]]:
    with _get_session(dsn) as db:
        row = db.get(EmSubject, subject_id)
        if row is None:
            return None
        for key, value in fields.items():
            setattr(row, key, value)
        row.updated_at = now()
        db.flush()
        return subject_dict(row)


def set_status(
    dsn: str, subject_id: str, status: str, *, reason: str = "", causes: Optional[list] = None
) -> Optional[dict[str, Any]]:
    with _get_session(dsn) as db:
        row = db.get(EmSubject, subject_id)
        if row is None:
            return None
        before = row.status
        if before == status:
            return subject_dict(row)
        row.status = status
        row.updated_at = now()
        db.add(
            EmSubjectEvent(
                id=str(uuid.uuid4()),
                subject_id=row.id,
                at=now(),
                kind="status_changed",
                diff={"before": before, "after": status, "reason": reason},
                cause_observation_ids=causes,
            )
        )
        db.flush()
        return subject_dict(row)


def get_subject(dsn: str, subject_id: str) -> Optional[dict[str, Any]]:
    with _get_session(dsn) as db:
        row = db.get(EmSubject, subject_id)
        return subject_dict(row) if row is not None else None


def find_subject(dsn: str, *, kind: str, slug: str) -> Optional[dict[str, Any]]:
    with _get_session(dsn) as db:
        row = db.scalars(
            select(EmSubject).where(EmSubject.kind == kind, EmSubject.slug == slug)
        ).first()
        return subject_dict(row) if row is not None else None


def list_subjects(
    dsn: str,
    *,
    kinds: Optional[Iterable[str]] = None,
    statuses: Optional[Iterable[str]] = None,
) -> list[dict[str, Any]]:
    with _get_session(dsn) as db:
        stmt = select(EmSubject)
        if kinds:
            stmt = stmt.where(EmSubject.kind.in_(list(kinds)))
        if statuses:
            stmt = stmt.where(EmSubject.status.in_(list(statuses)))
        rows = db.scalars(stmt.order_by(EmSubject.updated_at.desc())).all()
        return [subject_dict(r) for r in rows]


def add_event(
    dsn: str,
    subject_id: str,
    kind: str,
    diff: Optional[dict[str, Any]] = None,
    causes: Optional[list[str]] = None,
) -> None:
    with _get_session(dsn) as db:
        db.add(
            EmSubjectEvent(
                id=str(uuid.uuid4()),
                subject_id=subject_id,
                at=now(),
                kind=kind,
                diff=diff,
                cause_observation_ids=causes,
            )
        )


def events_since(dsn: str, since: datetime) -> list[dict[str, Any]]:
    with _get_session(dsn) as db:
        rows = db.execute(
            select(EmSubjectEvent, EmSubject.title, EmSubject.kind)
            .join(EmSubject, EmSubject.id == EmSubjectEvent.subject_id)
            .where(EmSubjectEvent.at >= since)
            .order_by(EmSubjectEvent.at)
        ).all()
        return [
            {
                "subject_id": ev.subject_id,
                "title": title,
                "subject_kind": skind,
                "kind": ev.kind,
                "at": ev.at,
                "diff": ev.diff or {},
            }
            for ev, title, skind in rows
        ]


def link(
    dsn: str,
    observation_id: str,
    subject_id: str,
    *,
    role: str = "mentions",
    score: Optional[float] = None,
    linked_by: str = "embedding",
    vector: Optional[list[float]] = None,
) -> bool:
    """Attach an observation; updates last_evidence_at and the running centroid.

    Returns False when the link already existed.
    """
    stamp = now()
    with _get_session(dsn) as db:
        exists = db.scalars(
            select(EmLink).where(
                EmLink.observation_id == observation_id, EmLink.subject_id == subject_id
            )
        ).first()
        if exists is not None:
            return False
        db.add(
            EmLink(
                id=str(uuid.uuid4()),
                observation_id=observation_id,
                subject_id=subject_id,
                role=role,
                score=score,
                linked_by=linked_by,
                created_at=stamp,
            )
        )
        row = db.get(EmSubject, subject_id)
        if row is not None:
            row.last_evidence_at = stamp
            row.updated_at = stamp
            vec = unit(vector)
            if vec is not None:
                fields = dict(row.fields or {})
                count = int(fields.get("_n") or 0)
                old = unit(row.centroid) if row.centroid is not None else None
                if old is None or old.shape != vec.shape:
                    merged = vec
                    count = 1
                else:
                    merged = (old * count + vec) / (count + 1)
                    count += 1
                row.centroid = [float(x) for x in merged]
                fields["_n"] = count
                row.fields = fields
        return True


def _obs_row(obs: EmObservation, source: Optional[EmSource]) -> dict[str, Any]:
    """Observation dict; ``created_at`` is when it was *said* when known.

    Meeting start (``em_sources.occurred_at``) plus the offset into the
    meeting; falls back to processing time.
    """
    said = obs.created_at
    if source is not None and source.occurred_at is not None:
        said = source.occurred_at + timedelta(seconds=float(obs.start_sec or 0.0))
    return {
        "id": obs.id,
        "type": obs.type,
        "text": obs.text,
        "context": obs.context or "",
        "quote": obs.quote or "",
        "speaker_name": obs.speaker_name or "",
        "start_sec": obs.start_sec,
        "fields": dict(obs.fields or {}),
        "status": obs.status,
        "created_at": said,
        "processed_at": obs.created_at,
        "source_ref": source.ref if source is not None else "",
        "source_kind": source.kind if source is not None else "",
        "embedding": [float(x) for x in obs.embedding] if obs.embedding is not None else None,
    }


def subject_observations(
    dsn: str, subject_id: str, *, since: Optional[datetime] = None, limit: int = 200
) -> list[dict[str, Any]]:
    """Linked observations (not rejected), newest last, with link role."""
    with _get_session(dsn) as db:
        stmt = (
            select(EmObservation, EmSource, EmLink.role, EmLink.created_at)
            .join(EmLink, EmLink.observation_id == EmObservation.id)
            .join(EmSource, EmSource.id == EmObservation.source_id, isouter=True)
            .where(EmLink.subject_id == subject_id, EmObservation.status != "rejected")
        )
        if since is not None:
            stmt = stmt.where(EmLink.created_at > since)
        rows = db.execute(stmt.order_by(EmLink.created_at.desc()).limit(limit)).all()
        out = []
        for obs, source, role, linked_at in reversed(rows):
            item = _obs_row(obs, source)
            item["role"] = role
            item["linked_at"] = linked_at
            out.append(item)
        return out


def link_counts(dsn: str, subject_id: str, *, days: int) -> int:
    since = now() - timedelta(days=days)
    with _get_session(dsn) as db:
        return int(
            db.scalar(
                select(func.count())
                .select_from(EmLink)
                .where(EmLink.subject_id == subject_id, EmLink.created_at >= since)
            )
            or 0
        )


def recent_observations(
    dsn: str,
    *,
    days: int = 30,
    unrouted: bool = False,
    types: Optional[Iterable[str]] = None,
    with_embedding: bool = True,
    limit: int = 2000,
) -> list[dict[str, Any]]:
    """Live observations from the last ``days`` (optionally with no link)."""
    since = now() - timedelta(days=days)
    with _get_session(dsn) as db:
        stmt = (
            select(EmObservation, EmSource)
            .join(EmSource, EmSource.id == EmObservation.source_id, isouter=True)
            .where(EmObservation.status == "live", EmObservation.created_at >= since)
        )
        if types:
            stmt = stmt.where(EmObservation.type.in_(list(types)))
        if with_embedding:
            stmt = stmt.where(EmObservation.embedding.is_not(None))
        if unrouted:
            linked = select(EmLink.observation_id)
            stmt = stmt.where(EmObservation.id.not_in(linked))
        rows = db.execute(stmt.order_by(EmObservation.created_at).limit(limit)).all()
        return [_obs_row(obs, source) for obs, source in rows]


def insert_observation(dsn: str, *, source_kind: str, source_ref: str, **fields: Any) -> str:
    """Add one observation to a source's latest run (used by the directed pass)."""
    with _get_session(dsn) as db:
        source = db.scalars(
            select(EmSource).where(EmSource.kind == source_kind, EmSource.ref == source_ref)
        ).first()
        if source is None:
            raise ValueError(f"no emergence source {source_kind}/{source_ref}")
        oid = fields.pop("id", None) or str(uuid.uuid4())
        db.add(
            EmObservation(
                id=oid,
                source_id=source.id,
                run_id=source.run_id or "",
                created_at=now(),
                status=fields.pop("status", "live"),
                **fields,
            )
        )
        return oid


def nearest(
    vector: Any, candidates: list[dict[str, Any]], *, key: str = "centroid"
) -> list[tuple[float, dict[str, Any]]]:
    """Candidates sorted by cosine distance to ``vector`` (closest first)."""
    scored = [
        (cosine_distance(vector, c.get(key)), c) for c in candidates if c.get(key) is not None
    ]
    scored.sort(key=lambda pair: pair[0])
    return scored
