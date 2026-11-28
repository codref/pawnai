"""PostgreSQL persistence for emergence sources, windows and observations."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional

from sqlalchemy import delete, select, update

from pawn_agent.utils.db import EmObservation, EmSource, EmWindow, _get_session

if TYPE_CHECKING:
    from pawn_agent.core.emergence.pipeline import ObserveResult


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def save_run(dsn: str, result: "ObserveResult", *, pipeline_version: str) -> str:
    """Store one run. Replaces windows; earlier observations become superseded.

    Returns the source id.
    """
    now = _now()
    with _get_session(dsn) as db:
        source = db.scalars(
            select(EmSource).where(
                EmSource.kind == result.source_kind, EmSource.ref == result.source_ref
            )
        ).first()
        if source is None:
            source = EmSource(id=result.run_id, kind=result.source_kind, ref=result.source_ref)
            db.add(source)
        source.content_hash = result.content_hash
        source.pipeline_version = pipeline_version
        source.run_id = result.run_id
        source.status = "done"
        source.stats = dict(result.stats)
        source.processed_at = now
        db.flush()
        db.execute(
            update(EmObservation)
            .where(
                EmObservation.source_id == source.id,
                EmObservation.status != "superseded",
            )
            .values(status="superseded", window_id=None)
        )
        db.execute(delete(EmWindow).where(EmWindow.source_id == source.id))
        for w in result.windows:
            db.add(
                EmWindow(
                    id=w.id,
                    source_id=source.id,
                    run_id=result.run_id,
                    ordinal=w.window.index,
                    char_start=w.window.char_start,
                    char_end=w.window.char_start + len(w.window.text),
                    start_sec=w.start_sec,
                    end_sec=w.end_sec,
                    content_hash=w.content_hash,
                    triage=dict(w.triage) if w.triage else None,
                    triage_model=result.model or None,
                    embedding=w.embedding,
                )
            )
        db.flush()
        for obs in result.observations:
            db.add(
                EmObservation(
                    id=obs["id"],
                    source_id=source.id,
                    window_id=obs.get("window_id"),
                    run_id=result.run_id,
                    type=obs["type"],
                    text=obs["text"],
                    context=obs.get("context") or None,
                    speaker_name=obs.get("speaker_name") or None,
                    quote=obs.get("quote") or None,
                    quote_score=obs.get("quote_score"),
                    start_sec=obs.get("start_sec"),
                    fields=obs.get("fields") or None,
                    confidence=obs.get("confidence"),
                    extractor=obs.get("extractor"),
                    model=obs.get("model") or None,
                    status=obs["status"],
                    reject_reason=obs.get("reject_reason"),
                    embedding=obs.get("embedding"),
                    created_at=now,
                )
            )
        return source.id


def observation_dict(row: EmObservation) -> dict[str, Any]:
    return {
        "id": row.id,
        "source_id": row.source_id,
        "run_id": row.run_id,
        "type": row.type,
        "text": row.text,
        "context": row.context or "",
        "speaker_name": row.speaker_name or "",
        "quote": row.quote or "",
        "quote_score": row.quote_score,
        "start_sec": row.start_sec,
        "fields": row.fields or {},
        "confidence": row.confidence,
        "extractor": row.extractor,
        "model": row.model,
        "status": row.status,
        "reject_reason": row.reject_reason,
        "created_at": row.created_at.isoformat() if row.created_at else None,
    }


def get_source(dsn: str, ref: str, kind: Optional[str] = None) -> Optional[dict[str, Any]]:
    with _get_session(dsn) as db:
        stmt = select(EmSource).where(EmSource.ref == ref)
        if kind:
            stmt = stmt.where(EmSource.kind == kind)
        row = db.scalars(stmt.order_by(EmSource.processed_at.desc())).first()
        if row is None:
            return None
        return {
            "id": row.id,
            "kind": row.kind,
            "ref": row.ref,
            "run_id": row.run_id,
            "stats": row.stats or {},
            "processed_at": row.processed_at.isoformat() if row.processed_at else None,
        }


def list_observations(
    dsn: str,
    ref: str,
    *,
    kind: Optional[str] = None,
    statuses: tuple[str, ...] = ("live",),
) -> list[dict[str, Any]]:
    """Observations of the latest run for a source ref, in time order."""
    source = get_source(dsn, ref, kind)
    if source is None:
        return []
    with _get_session(dsn) as db:
        rows = db.scalars(
            select(EmObservation)
            .where(
                EmObservation.source_id == source["id"],
                EmObservation.run_id == source["run_id"],
                EmObservation.status.in_(statuses),
            )
            .order_by(EmObservation.start_sec.asc().nulls_last(), EmObservation.created_at)
        ).all()
        return [observation_dict(r) for r in rows]
