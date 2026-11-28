"""Speakers gallery HTTP helpers.

List and detail stay read-only over the gallery tables. Rename, merge and
deactivate go through :class:`SpeakerGallery` so the portal and the CLI share
one write path.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import func, select
from sqlalchemy.orm import Session as OrmSession

from pawn_diarize.core.database import get_engine
from pawn_diarize.core.speaker_gallery import SpeakerGallery
from pawn_core.database import SessionAnalysis, SessionSpeakerMap, SpeakerEnrollment

logger = logging.getLogger(__name__)


class SpeakersApiError(Exception):
    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _gallery(cfg: Any) -> SpeakerGallery:
    return SpeakerGallery(cfg.db_dsn, config=getattr(cfg, "speakers", None))


def _iso(value: Any) -> Optional[str]:
    if isinstance(value, datetime):
        return value.isoformat()
    return None


def _people_path(cfg: Any, speaker_id: str) -> Optional[str]:
    """Vault key of the People note, when that file exists."""
    try:
        from pawn_agent.core.people.notes import people_note_key  # noqa: PLC0415
        from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

        key = people_note_key(cfg, speaker_id)
        if vault_store_from_config(cfg).exists(key):
            return key
    except Exception as exc:  # noqa: BLE001
        logger.debug("people path %s: %s", speaker_id, exc)
    return None


def _stats(cfg: Any) -> dict[str, dict[str, Any]]:
    """Enrollment totals and the latest mapped session for every speaker."""
    out: dict[str, dict[str, Any]] = {}
    try:
        engine = get_engine(cfg.db_dsn)
    except Exception as exc:  # noqa: BLE001
        logger.debug("speaker stats engine: %s", exc)
        return out
    try:
        with OrmSession(engine) as db:
            for sid, count, seconds in db.execute(
                select(
                    SpeakerEnrollment.speaker_id,
                    func.count(),
                    func.coalesce(func.sum(SpeakerEnrollment.duration), 0.0),
                ).group_by(SpeakerEnrollment.speaker_id)
            ):
                bucket = out.setdefault(str(sid), {})
                bucket["enrollments"] = int(count or 0)
                bucket["seconds"] = float(seconds or 0)
            sessions: dict[str, set[str]] = {}
            latest: dict[str, tuple[datetime, str]] = {}
            for sid, session_id, updated in db.execute(
                select(
                    SessionSpeakerMap.speaker_id,
                    SessionSpeakerMap.session_id,
                    SessionSpeakerMap.updated_at,
                ).where(SessionSpeakerMap.speaker_id.is_not(None))
            ):
                key = str(sid)
                sessions.setdefault(key, set()).add(str(session_id))
                if isinstance(updated, datetime):
                    prev = latest.get(key)
                    if prev is None or updated >= prev[0]:
                        latest[key] = (updated, str(session_id))
            for key, ids in sessions.items():
                bucket = out.setdefault(key, {})
                bucket["sessions"] = len(ids)
                hit = latest.get(key)
                if hit:
                    bucket["last_seen_at"] = hit[0]
                    bucket["last_seen_session"] = hit[1]
    except Exception as exc:  # noqa: BLE001
        logger.debug("speaker stats: %s", exc)
    return out


def _summary(cfg: Any, row: Any, stats: dict[str, dict[str, Any]]) -> dict[str, Any]:
    info = stats.get(str(row.id), {})
    seen = info.get("last_seen_at")
    return {
        "id": row.id,
        "display_name": row.display_name,
        "aliases": list(row.aliases or []),
        "active": bool(row.active),
        "enrollments": int(info.get("enrollments") or 0),
        "seconds": float(info.get("seconds") or 0),
        "sessions": int(info.get("sessions") or 0),
        "last_seen_session": info.get("last_seen_session"),
        "last_seen_at": _iso(seen),
        "people_path": _people_path(cfg, row.id),
        "notes": row.notes,
    }


def _matches(row: Any, query: str) -> bool:
    needle = query.strip().lower()
    if not needle:
        return True
    hay = " ".join(
        [str(row.id), str(row.display_name or "")] + [str(item) for item in (row.aliases or [])]
    )
    return needle in hay.lower()


def list_speakers(
    cfg: Any,
    *,
    query: str = "",
    include_inactive: bool = False,
) -> dict[str, Any]:
    gallery = _gallery(cfg)
    stats = _stats(cfg)
    data = [
        _summary(cfg, row, stats)
        for row in gallery.list_speakers(include_inactive=include_inactive)
        if _matches(row, query)
    ]
    return {"data": data}


def _enrollments(cfg: Any, speaker_id: str) -> list[dict[str, Any]]:
    gallery = _gallery(cfg)
    rows = []
    for row in gallery.list_enrollments(speaker_id):
        rows.append(
            {
                "id": row.id,
                "source_session_id": row.source_session_id,
                "start_time": row.start_time,
                "end_time": row.end_time,
                "duration": float(row.duration or 0),
                "quality_score": row.quality_score,
                "approved_at": _iso(row.approved_at),
            }
        )
    return rows


def _titles(cfg: Any, session_ids: list[str]) -> dict[str, str]:
    if not session_ids:
        return {}
    try:
        engine = get_engine(cfg.db_dsn)
        with OrmSession(engine) as db:
            found = db.execute(
                select(SessionAnalysis.session_id, SessionAnalysis.title).where(
                    SessionAnalysis.session_id.in_(session_ids)
                )
            ).all()
    except Exception as exc:  # noqa: BLE001
        logger.debug("speaker session titles: %s", exc)
        return {}
    return {str(sid): str(title) for sid, title in found if sid and title}


def _recent_sessions(cfg: Any, speaker_id: str, *, limit: int = 12) -> list[dict[str, Any]]:
    try:
        engine = get_engine(cfg.db_dsn)
        with OrmSession(engine) as db:
            rows = db.execute(
                select(SessionSpeakerMap.session_id, SessionSpeakerMap.updated_at)
                .where(SessionSpeakerMap.speaker_id == speaker_id)
                .order_by(SessionSpeakerMap.updated_at.desc())
            ).all()
    except Exception as exc:  # noqa: BLE001
        logger.debug("speaker sessions %s: %s", speaker_id, exc)
        return []
    ordered: list[str] = []
    seen_at: dict[str, Any] = {}
    for session_id, updated in rows:
        sid = str(session_id)
        if sid in seen_at:
            continue
        seen_at[sid] = updated
        ordered.append(sid)
        if len(ordered) >= limit:
            break
    titles = _titles(cfg, ordered)
    return [
        {
            "session_id": sid,
            "title": titles.get(sid) or sid,
            "created_at": _iso(seen_at.get(sid)),
        }
        for sid in ordered
    ]


def _require(cfg: Any, speaker_id: str) -> Any:
    row = _gallery(cfg).get_speaker(speaker_id)
    if row is None:
        raise SpeakersApiError(f"Unknown speaker: {speaker_id}", status_code=404)
    return row


def speaker_detail(cfg: Any, speaker_id: str) -> dict[str, Any]:
    row = _require(cfg, speaker_id)
    payload = _summary(cfg, row, _stats(cfg))
    payload["enrollment_rows"] = _enrollments(cfg, speaker_id)
    payload["recent_sessions"] = _recent_sessions(cfg, speaker_id)
    return payload


def _mapped_sessions(cfg: Any, speaker_id: str) -> list[str]:
    try:
        engine = get_engine(cfg.db_dsn)
        with OrmSession(engine) as db:
            rows = db.execute(
                select(SessionSpeakerMap.session_id)
                .where(SessionSpeakerMap.speaker_id == speaker_id)
                .distinct()
            ).all()
    except Exception as exc:  # noqa: BLE001
        logger.debug("mapped sessions %s: %s", speaker_id, exc)
        return []
    return [str(sid) for (sid,) in rows if sid]


def update_speaker(cfg: Any, speaker_id: str, **fields: Any) -> dict[str, Any]:
    """Patch the gallery card. A rename reports sessions still on the old name."""
    _require(cfg, speaker_id)
    display_name = fields.get("display_name") if "display_name" in fields else None
    if "display_name" in fields and not str(display_name or "").strip():
        raise SpeakersApiError("display_name is required", status_code=422)
    aliases = fields.get("aliases") if "aliases" in fields else None
    notes = fields.get("notes") if "notes" in fields else None
    try:
        _gallery(cfg).update_speaker(
            speaker_id,
            display_name=str(display_name).strip() if display_name is not None else None,
            aliases=aliases,
            notes=None if notes is None else str(notes),
        )
    except ValueError as exc:
        raise SpeakersApiError(str(exc), status_code=404) from exc
    payload = speaker_detail(cfg, speaker_id)
    mapped = _mapped_sessions(cfg, speaker_id)
    payload["referenced_sessions"] = len(mapped)
    payload["job_id"] = None
    payload["_reidentify_ids"] = []
    if fields.get("reidentify") and mapped:
        from pawn_server.core.sessions_api import queue_bulk_reidentify  # noqa: PLC0415

        queued = queue_bulk_reidentify(cfg, session_ids=mapped, limit=50)
        payload["job_id"] = queued["job_id"]
        payload["_reidentify_ids"] = list(queued["session_ids"])
    return payload


def deactivate_speaker(cfg: Any, speaker_id: str) -> dict[str, Any]:
    try:
        _gallery(cfg).deactivate_speaker(speaker_id)
    except ValueError as exc:
        raise SpeakersApiError(str(exc), status_code=404) from exc
    summary = _summary(cfg, _require(cfg, speaker_id), _stats(cfg))
    summary.pop("enrollment_rows", None)
    summary.pop("recent_sessions", None)
    return summary


def delete_enrollment(cfg: Any, speaker_id: str, enrollment_id: str) -> None:
    _require(cfg, speaker_id)
    gallery = _gallery(cfg)
    owned = {row.id for row in gallery.list_enrollments(speaker_id)}
    if enrollment_id not in owned:
        raise SpeakersApiError(f"Unknown enrollment: {enrollment_id}", status_code=404)
    try:
        gallery.remove_enrollment(enrollment_id)
    except ValueError as exc:
        raise SpeakersApiError(str(exc), status_code=404) from exc


def merge_speakers(
    cfg: Any,
    speaker_id: str,
    into_id: str,
    *,
    reidentify: bool = False,
) -> dict[str, Any]:
    try:
        _dst, affected = _gallery(cfg).merge_speakers(speaker_id, into_id)
    except ValueError as exc:
        text = str(exc)
        status = 422 if "itself" in text or "required" in text else 404
        raise SpeakersApiError(text, status_code=status) from exc
    result: dict[str, Any] = {
        "into": into_id,
        "speaker": speaker_detail(cfg, into_id),
        "affected_sessions": affected,
        "job_id": None,
        "_reidentify_ids": [],
    }
    if reidentify and affected:
        from pawn_server.core.sessions_api import queue_bulk_reidentify  # noqa: PLC0415

        queued = queue_bulk_reidentify(cfg, session_ids=affected, limit=50)
        result["job_id"] = queued["job_id"]
        result["_reidentify_ids"] = list(queued["session_ids"])
    return result


def ensure_people_note(cfg: Any, speaker_id: str) -> dict[str, Any]:
    """Create the People note when it is missing. An existing note is kept."""
    row = _require(cfg, speaker_id)
    try:
        from pawn_agent.core.people.notes import ensure_person_note  # noqa: PLC0415
        from pawn_server.core.vault_events import publish_vault_event  # noqa: PLC0415

        key, created = ensure_person_note(
            cfg,
            speaker_id=row.id,
            display_name=row.display_name,
            aliases=list(row.aliases or []),
        )
    except Exception as exc:  # noqa: BLE001
        raise SpeakersApiError(f"Could not write the people note: {exc}", status_code=422) from exc
    if created:
        publish_vault_event([key], source="speakers")
    return {"path": key, "created": created}
