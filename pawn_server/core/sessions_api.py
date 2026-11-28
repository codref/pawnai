"""HTTP helpers for Sessions management (list, detail, speaker train).

Wraps Speakers gallery + reidentify/relabel without loading heavy ML for
list/detail. Session enroll uses ``session_state`` centroids (no audio).
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import re
import shutil
import struct
import subprocess
import threading
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import quote

from sqlalchemy import func, select
from sqlalchemy.orm import Session as OrmSession

from pawn_agent.tools.list_sessions import list_session_candidates_impl
from pawn_core.database import (
    SessionAnalysis,
    SessionSpeakerMap,
    TranscriptionSegment,
    get_engine,
)
from pawn_core.vault_db import get_vault_note
from pawn_diarize.core.database import (
    SessionState,
    init_db,
    load_session_state,
    set_session_processing,
)
from pawn_diarize.core.speaker_gallery import SpeakerGallery

logger = logging.getLogger(__name__)

_ANON_SPEAKER_RE = re.compile(r"^SPEAKER_\d+$")
_SAMPLE_CHARS = 160
_TOPIC_SPLIT_RE = re.compile(r"[\n,;•]+")


class SessionsApiError(Exception):
    """Domain error with an HTTP status code for the API layer."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int = 422,
        headers: Optional[dict[str, str]] = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.headers = headers or {}


_AUDIO_MIME = {
    ".wav": "audio/wav",
    ".flac": "audio/flac",
    ".webm": "audio/webm",
    ".ogg": "audio/ogg",
    ".oga": "audio/ogg",
    ".opus": "audio/ogg",
    ".mp3": "audio/mpeg",
    ".aac": "audio/aac",
    ".m4a": "audio/mp4",
    ".mp4": "audio/mp4",
}
# Bytes read from the start of an object to sniff its type and duration.
_HEADER_BYTES = 8 * 1024
# ``session_files`` results, reused by the many Range requests of one player.
_FILES_TTL = 60.0
_FILES_CACHE: dict[str, tuple[float, list[dict[str, Any]]]] = {}
_FILES_LOCK = threading.Lock()
_AUDIO_URL_TTL = 6 * 60 * 60
_LIST_FETCH = 2000
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_anon(label: str) -> bool:
    return bool(_ANON_SPEAKER_RE.match(str(label or "").strip()))


def _transcript_path(cfg: Any, session_id: str) -> Optional[str]:
    try:
        row = get_vault_note(cfg.db_dsn, session_id)
        if row is not None and row.key:
            return str(row.key)
    except Exception as exc:  # noqa: BLE001
        logger.debug("vault_notes lookup %s failed: %s", session_id, exc)
    return None


def _label_stats(db_dsn: str, session_ids: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """session_id → local_label → {duration_s, segment_count}.

    Matches the vault Speakers table: only segments with non-empty transcript
    text, and talk time as the sum of turn durations (not first→last span).
    """
    if not session_ids:
        return {}
    engine = get_engine(db_dsn)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        with OrmSession(engine) as db:
            # SUM(end - start) per label; exclude blank ASR turns (noise clusters
            # that vault Speakers already drops).
            talk = TranscriptionSegment.end_time - TranscriptionSegment.start_time
            rows = db.execute(
                select(
                    TranscriptionSegment.session_id,
                    TranscriptionSegment.original_speaker_label,
                    func.coalesce(func.sum(talk), 0.0).label("talk_s"),
                    func.count().label("n"),
                )
                .where(TranscriptionSegment.session_id.in_(session_ids))
                .where(TranscriptionSegment.text.is_not(None))
                .where(func.length(func.trim(TranscriptionSegment.text)) > 0)
                .group_by(
                    TranscriptionSegment.session_id,
                    TranscriptionSegment.original_speaker_label,
                )
            ).all()
            for row in rows:
                sid = str(row.session_id)
                label = str(row.original_speaker_label or "SPEAKER_00")
                out.setdefault(sid, {})[label] = {
                    "duration_s": max(0.0, float(row.talk_s or 0.0)),
                    "segment_count": int(row.n or 0),
                }
    except Exception as exc:  # noqa: BLE001
        logger.debug("label stats failed: %s", exc)
    return out


def _session_maps(db_dsn: str, session_ids: list[str]) -> dict[str, dict[str, SessionSpeakerMap]]:
    if not session_ids:
        return {}
    engine = get_engine(db_dsn)
    out: dict[str, dict[str, SessionSpeakerMap]] = {}
    try:
        with OrmSession(engine) as db:
            rows = list(
                db.scalars(
                    select(SessionSpeakerMap).where(SessionSpeakerMap.session_id.in_(session_ids))
                )
            )
            for row in rows:
                db.expunge(row)
                out.setdefault(str(row.session_id), {})[str(row.local_label)] = row
    except Exception as exc:  # noqa: BLE001
        logger.debug("session maps failed: %s", exc)
    return out


def _speakers_trained_flags(db_dsn: str, session_ids: list[str]) -> dict[str, bool]:
    """session_id → True when speakers_trained_at is set."""
    if not session_ids:
        return {}
    engine = get_engine(db_dsn)
    out: dict[str, bool] = {}
    try:
        with OrmSession(engine) as db:
            rows = db.execute(
                select(SessionState.session_id, SessionState.speakers_trained_at).where(
                    SessionState.session_id.in_(session_ids)
                )
            ).all()
            for row in rows:
                out[str(row.session_id)] = row.speakers_trained_at is not None
    except Exception as exc:  # noqa: BLE001
        logger.debug("speakers_trained flags failed: %s", exc)
    return out


def _is_ignored(mapped: Optional[SessionSpeakerMap]) -> bool:
    return bool(mapped and str(getattr(mapped, "match_method", "") or "") == "ignore")


def _speakers_summary(
    labels: dict[str, dict[str, Any]],
    mapping: dict[str, SessionSpeakerMap],
    *,
    speakers_trained: bool = False,
) -> tuple[list[str], int, bool]:
    """Return (speaker pills, unmapped_count, needs_speakers).

    Anonymous labels marked ``match_method=ignore`` are omitted from the
    needs-training signal (noise). When *speakers_trained* is true, needs is
    always false.
    """
    pills: list[str] = []
    unmapped = 0
    for label in sorted(labels.keys()):
        mapped = mapping.get(label)
        if _is_ignored(mapped):
            # Omit ignored noise from card pills (Train still lists them).
            continue
        if mapped and mapped.display_name:
            name = str(mapped.display_name)
            if name not in pills:
                pills.append(name)
            if _is_anon(label) and not mapped.speaker_id:
                unmapped += 1
        elif mapped and mapped.speaker_id:
            name = str(mapped.display_name or mapped.speaker_id)
            if name not in pills:
                pills.append(name)
        elif _is_anon(label):
            unmapped += 1
            if label not in pills:
                pills.append(label)
        else:
            if label not in pills:
                pills.append(label)

    needs = False
    if not speakers_trained:
        for label in labels:
            if not _is_anon(label):
                continue
            mapped = mapping.get(label)
            if _is_ignored(mapped):
                continue
            if mapped and mapped.speaker_id:
                continue
            needs = True
            break

    anon_count = sum(
        1 for label in labels if _is_anon(label) and not _is_ignored(mapping.get(label))
    )
    return pills, anon_count if anon_count else unmapped, needs


def _parse_topics(raw: Optional[str]) -> list[str]:
    if not raw or not str(raw).strip():
        return []
    out: list[str] = []
    for part in _TOPIC_SPLIT_RE.split(str(raw)):
        topic = part.strip().lstrip("-* ").strip()
        if topic and topic not in out:
            out.append(topic)
        if len(out) >= 8:
            break
    return out


def _parse_tags(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(t).strip() for t in raw if str(t).strip()][:12]
    if isinstance(raw, str) and raw.strip():
        return _parse_topics(raw)[:12]
    return []


def _load_analyses(db_dsn: str, session_ids: list[str]) -> dict[str, SessionAnalysis]:
    if not session_ids:
        return {}
    engine = get_engine(db_dsn)
    out: dict[str, SessionAnalysis] = {}
    try:
        with OrmSession(engine) as db:
            rows = list(
                db.scalars(
                    select(SessionAnalysis)
                    .where(SessionAnalysis.session_id.in_(session_ids))
                    .order_by(
                        SessionAnalysis.session_id,
                        SessionAnalysis.analyzed_at.desc(),
                    )
                )
            )
            for row in rows:
                sid = str(row.session_id or "")
                if not sid or sid in out:
                    continue
                db.expunge(row)
                out[sid] = row
    except Exception as exc:  # noqa: BLE001
        logger.debug("session analyses load failed: %s", exc)
    return out


def _row_matches_query(row: dict[str, Any], q: str) -> bool:
    needle = q.strip().lower()
    if not needle:
        return True
    haystacks = [
        str(row.get("session_id") or ""),
        str(row.get("title") or ""),
        str(row.get("summary") or ""),
        str(row.get("sentiment") or ""),
        " ".join(row.get("speakers") or []),
        " ".join(row.get("tags") or []),
        " ".join(row.get("topics") or []),
    ]
    return any(needle in h.lower() for h in haystacks)


def _day_key(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, datetime):
        return value.date().isoformat()
    text = str(value).strip()
    return text[:10] if len(text) >= 10 and text[4:5] == "-" else ""


def _require_date(value: str, name: str) -> str:
    text = (value or "").strip()
    if text and not _DATE_RE.match(text):
        raise SessionsApiError(f"{name} must be YYYY-MM-DD", status_code=422)
    return text


def list_sessions_page(
    cfg: Any,
    *,
    limit: int = 50,
    offset: int = 0,
    q: str = "",
    needs_speakers: bool = False,
    speaker: str = "",
    date_from: str = "",
    date_to: str = "",
    sort: str = "recent",
    has_analysis: Optional[bool] = None,
) -> dict[str, Any]:
    """Recent sessions plus speaker facets for the Sessions tab."""
    sort_key = (sort or "recent").strip().lower()
    if sort_key not in ("recent", "oldest", "duration"):
        raise SessionsApiError("sort must be recent, oldest, or duration", status_code=422)
    start_day = _require_date(date_from, "from")
    end_day = _require_date(date_to, "to")
    page_limit = max(1, min(int(limit or 50), 200))
    page_offset = max(0, int(offset or 0))
    # Broad fetch; filter q/speaker/dates in Python so search hits title/tags.
    fetch_limit = _LIST_FETCH
    candidates = list_session_candidates_impl(cfg, name_filter="", limit=fetch_limit)
    session_ids = [c.session_id for c in candidates]
    label_stats = _label_stats(cfg.db_dsn, session_ids)
    maps = _session_maps(cfg.db_dsn, session_ids)
    trained_flags = _speakers_trained_flags(cfg.db_dsn, session_ids)
    analyses = _load_analyses(cfg.db_dsn, session_ids)

    enriched: list[dict[str, Any]] = []
    speaker_counts: dict[str, int] = {}
    tag_counts: dict[str, int] = {}
    needs_total = 0
    for cand in candidates:
        labels = label_stats.get(cand.session_id, {})
        mapping = maps.get(cand.session_id, {})
        trained = bool(trained_flags.get(cand.session_id))
        speakers, unmapped_count, needs = _speakers_summary(
            labels, mapping, speakers_trained=trained
        )
        analysis = analyses.get(cand.session_id)
        tags = _parse_tags(getattr(analysis, "tags", None) if analysis else None)
        topics = _parse_topics(getattr(analysis, "key_topics", None) if analysis else None)
        sentiment = None
        if analysis and getattr(analysis, "sentiment", None):
            sentiment = str(analysis.sentiment).strip() or None
        title = cand.title or (analysis.title if analysis else None) or cand.session_id
        summary = cand.summary or (analysis.summary if analysis else None)
        row = {
            "session_id": cand.session_id,
            "title": title,
            "updated_at": (
                cand.updated_at.isoformat()
                if getattr(cand, "updated_at", None) is not None
                else None
            ),
            "created_at": (
                cand.created_at.isoformat()
                if getattr(cand, "created_at", None) is not None
                else None
            ),
            "segments": cand.segments,
            "duration_seconds": cand.duration_seconds,
            "summary": summary,
            "transcript_path": None,
            "speakers": speakers,
            "unmapped_count": unmapped_count,
            "needs_speakers": needs,
            "speakers_trained": trained,
            "tags": tags,
            "topics": topics,
            "sentiment": sentiment,
            "has_analysis": analysis is not None,
        }
        enriched.append(row)
        if needs:
            needs_total += 1
        for name in speakers:
            if _is_anon(name):
                continue
            speaker_counts[name] = speaker_counts.get(name, 0) + 1
        for tag in tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

    facets_speakers = _top_counts(speaker_counts, 12)
    facets_tags = _top_counts(tag_counts, 20)

    speaker_filter = (speaker or "").strip().lower()
    matched: list[dict[str, Any]] = []
    for row in enriched:
        if needs_speakers and not row["needs_speakers"]:
            continue
        if speaker_filter:
            names = [str(s).lower() for s in (row.get("speakers") or [])]
            if speaker_filter not in names:
                continue
        if has_analysis is True and not row["has_analysis"]:
            continue
        if has_analysis is False and row["has_analysis"]:
            continue
        day = _day_key(row.get("created_at"))
        if start_day or end_day:
            if not day:
                continue
            if start_day and day < start_day:
                continue
            if end_day and day > end_day:
                continue
        if q.strip() and not _row_matches_query(row, q):
            continue
        matched.append(row)

    if sort_key == "oldest":
        matched.sort(key=lambda row: (row.get("created_at") or "9999", row["session_id"]))
    elif sort_key == "duration":
        matched.sort(
            key=lambda row: (-(float(row.get("duration_seconds") or 0.0)), row["session_id"])
        )
    else:
        matched.sort(key=lambda row: (row.get("updated_at") or "", row["session_id"]), reverse=True)

    page = matched[page_offset : page_offset + page_limit]
    page_ids = [str(row["session_id"]) for row in page]
    open_counts = _open_item_counts(cfg.db_dsn, page_ids)
    events = _events_by_session(cfg, set(page_ids))
    for row in page:
        sid = str(row["session_id"])
        row["transcript_path"] = _transcript_path(cfg, sid)
        row["items_open"] = int(open_counts.get(sid, 0))
        row["event"] = events.get(sid)

    return {
        "data": page,
        "facets": {
            "speakers": facets_speakers,
            "needs_speakers": needs_total,
            "tags": facets_tags,
            "total": len(matched),
        },
    }


def _top_counts(counts: dict[str, int], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))
    return [{"name": name, "count": count} for name, count in ranked[:limit]]


def _open_item_counts(db_dsn: str, session_ids: list[str]) -> dict[str, int]:
    """session_id → open coworker items whose source_ref is that session."""
    if not session_ids:
        return {}
    try:
        from pawn_agent.core.coworker.db import OPEN_STATUSES  # noqa: PLC0415
        from pawn_agent.utils.db import CoworkerItem  # noqa: PLC0415

        engine = get_engine(db_dsn)
        with OrmSession(engine) as db:
            rows = db.execute(
                select(CoworkerItem.source_ref, func.count())
                .where(CoworkerItem.source_ref.in_(session_ids))
                .where(CoworkerItem.status.in_(tuple(OPEN_STATUSES)))
                .group_by(CoworkerItem.source_ref)
            ).all()
        return {str(row[0]): int(row[1]) for row in rows}
    except Exception as exc:  # noqa: BLE001
        logger.debug("open item counts failed: %s", exc)
        return {}


def _events_by_session(cfg: Any, session_ids: set[str]) -> dict[str, dict[str, str]]:
    """session_id → {id, title} for the earliest event note that links it."""
    if not session_ids:
        return {}
    try:
        from pawn_server.core.planner import list_events  # noqa: PLC0415

        events = list_events(cfg)
    except Exception as exc:  # noqa: BLE001
        logger.debug("event link lookup failed: %s", exc)
        return {}
    out: dict[str, dict[str, str]] = {}
    for event in events:
        sid = str(event.get("session_id") or "").strip()
        if not sid or sid not in session_ids or sid in out:
            continue
        out[sid] = {
            "id": str(event.get("id") or ""),
            "title": str(event.get("title") or ""),
        }
    return out


def list_sessions(
    cfg: Any,
    *,
    limit: int = 20,
    q: str = "",
    needs_speakers: bool = False,
    speaker: str = "",
) -> list[dict[str, Any]]:
    """Backward-compatible list (capture picker + older callers)."""
    return list_sessions_page(
        cfg,
        limit=limit,
        q=q,
        needs_speakers=needs_speakers,
        speaker=speaker,
    )["data"]


def _sample_text(db_dsn: str, session_id: str, local_label: str) -> str:
    engine = get_engine(db_dsn)
    with OrmSession(engine) as db:
        rows = list(
            db.scalars(
                select(TranscriptionSegment)
                .where(TranscriptionSegment.session_id == session_id)
                .where(TranscriptionSegment.original_speaker_label == local_label)
                .order_by(TranscriptionSegment.start_time)
                .limit(8)
            )
        )
    parts = [str(r.text or "").strip() for r in rows if str(r.text or "").strip()]
    text = " ".join(parts).strip()
    if len(text) > _SAMPLE_CHARS:
        return text[: _SAMPLE_CHARS - 1].rstrip() + "…"
    return text


def get_session(cfg: Any, session_id: str) -> dict[str, Any]:
    """Detail payload for the Train modal."""
    sid = (session_id or "").strip()
    if not sid:
        raise SessionsApiError("session_id is required", status_code=422)

    candidates = list_session_candidates_impl(cfg, name_filter=sid, limit=50)
    cand = next((c for c in candidates if c.session_id == sid), None)
    if cand is None:
        # Still allow detail when segments exist but name filter missed.
        stats = _label_stats(cfg.db_dsn, [sid])
        if sid not in stats:
            raise SessionsApiError(f"Unknown session: {sid}", status_code=404)
        title = sid
        segments = sum(v.get("segment_count", 0) for v in stats[sid].values())
        duration = None
        summary = None
        updated_at = None
        created_at = None
        labels_map = stats[sid]
    else:
        title = cand.title or sid
        segments = cand.segments
        duration = cand.duration_seconds
        summary = cand.summary
        updated_at = cand.updated_at.isoformat() if cand.updated_at is not None else None
        created_at = cand.created_at.isoformat() if cand.created_at is not None else None
        labels_map = _label_stats(cfg.db_dsn, [sid]).get(sid, {})

    gallery = SpeakerGallery(cfg.db_dsn, config=getattr(cfg, "speakers", None))
    mapping = gallery.load_session_map(sid)
    engine = get_engine(cfg.db_dsn)
    init_db(engine)
    prior, _, _, _ = load_session_state(sid, engine)
    trained = bool(_speakers_trained_flags(cfg.db_dsn, [sid]).get(sid))
    segment_rows = _load_segments(cfg.db_dsn, sid)
    samples_by_label = _samples_by_label(segment_rows)
    talk_total = sum(float(info.get("duration_s") or 0.0) for info in labels_map.values())

    labels_out: list[dict[str, Any]] = []
    for label in sorted(labels_map.keys()):
        info = labels_map[label]
        mapped = mapping.get(label)
        centroid = None
        if isinstance(prior, dict):
            entry = prior.get(label)
            if isinstance(entry, dict):
                centroid = entry.get("embedding")
        talk = float(info.get("duration_s") or 0.0)
        ignored = _is_ignored(mapped)
        mapped_id = str(mapped.speaker_id) if mapped and mapped.speaker_id else None
        suggested_id = None
        suggested_score = None
        if centroid is not None and not ignored and not mapped_id:
            suggested_id, suggested_score = _suggest_speaker(gallery, centroid)
        share = (talk / talk_total) if talk_total > 0 else 0.0
        labels_out.append(
            {
                "local_label": label,
                "duration_s": talk,
                "sample_text": _sample_text(cfg.db_dsn, sid, label),
                "mapped_speaker_id": mapped_id,
                "mapped_name": (
                    str(mapped.display_name) if mapped and mapped.display_name else None
                ),
                "ignored": ignored,
                "has_centroid": centroid is not None,
                "samples": samples_by_label.get(label, []),
                "talk_seconds": talk,
                "share": round(share, 4),
                "suggested_speaker_id": suggested_id,
                "suggested_score": suggested_score,
            }
        )

    speakers_summary, unmapped_count, needs = _speakers_summary(
        labels_map, mapping, speakers_trained=trained
    )
    analyses = _load_analyses(cfg.db_dsn, [sid])
    analysis_row = analyses.get(sid)
    if analysis_row is not None and not summary:
        summary = analysis_row.summary
    if analysis_row is not None and title == sid and analysis_row.title:
        title = analysis_row.title
    open_counts = _open_item_counts(cfg.db_dsn, [sid])
    linked = _events_by_session(cfg, {sid}).get(sid)
    gallery_out: list[dict[str, Any]] = []
    for sp in gallery.list_speakers():
        enrollments = gallery.list_enrollments(sp.id)
        gallery_out.append(
            {
                "id": sp.id,
                "display_name": sp.display_name,
                "enrollments": len(enrollments),
            }
        )

    return {
        "session_id": sid,
        "title": title,
        "updated_at": updated_at,
        "created_at": created_at,
        "segments": segments,
        "duration_seconds": duration,
        "summary": summary,
        "transcript_path": _transcript_path(cfg, sid),
        "speakers": speakers_summary,
        "unmapped_count": unmapped_count,
        "needs_speakers": needs,
        "speakers_trained": trained,
        "tags": _parse_tags(getattr(analysis_row, "tags", None) if analysis_row else None),
        "topics": _parse_topics(
            getattr(analysis_row, "key_topics", None) if analysis_row else None
        ),
        "sentiment": (
            str(analysis_row.sentiment).strip()
            if analysis_row and getattr(analysis_row, "sentiment", None)
            else None
        ),
        "has_analysis": analysis_row is not None,
        "items_open": int(open_counts.get(sid, 0)),
        "event": linked,
        "labels": labels_out,
        "gallery": gallery_out,
        "analysis": _analysis_view(analysis_row),
        "analysis_path": _analysis_path(cfg, sid),
        "side_chats": _side_chats(cfg, sid),
        "processing": _is_processing(cfg.db_dsn, sid),
    }


def create_speaker(cfg: Any, display_name: str) -> dict[str, Any]:
    name = (display_name or "").strip()
    if not name:
        raise SessionsApiError("display_name is required", status_code=422)
    gallery = SpeakerGallery(cfg.db_dsn, config=getattr(cfg, "speakers", None))
    existing = gallery.find_speaker_by_name(name)
    if existing is not None:
        enrollments = gallery.list_enrollments(existing.id)
        return {
            "id": existing.id,
            "display_name": existing.display_name,
            "enrollments": len(enrollments),
            "created": False,
        }
    try:
        sp = gallery.create_speaker(name)
    except ValueError as exc:
        raise SessionsApiError(str(exc), status_code=422) from exc
    return {
        "id": sp.id,
        "display_name": sp.display_name,
        "enrollments": 0,
        "created": True,
    }


def enroll_session_speaker(
    cfg: Any,
    session_id: str,
    *,
    speaker: str,
    from_label: str,
    force: bool = False,
) -> dict[str, Any]:
    """Approve a voiceprint from a session label centroid."""
    import numpy as np

    sid = (session_id or "").strip()
    label = (from_label or "").strip()
    speaker_key = (speaker or "").strip()
    if not sid or not label or not speaker_key:
        raise SessionsApiError("session_id, speaker, and from_label are required", status_code=422)

    gallery = SpeakerGallery(cfg.db_dsn, config=getattr(cfg, "speakers", None))
    sp = gallery.get_speaker(speaker_key) or gallery.find_speaker_by_name(speaker_key)
    if sp is None:
        sp = gallery.create_speaker(speaker_key)

    engine = get_engine(cfg.db_dsn)
    init_db(engine)
    prior, _, _, _ = load_session_state(sid, engine)
    info = (prior or {}).get(label)
    if not info or not isinstance(info, dict) or info.get("embedding") is None:
        raise SessionsApiError(
            f"No session_state embedding for label {label!r}",
            status_code=422,
        )
    embedding = np.asarray(info["embedding"], dtype="float32")

    source_audio = None
    start: Optional[float] = None
    end: Optional[float] = None
    with OrmSession(engine) as db:
        rows = list(
            db.scalars(
                select(TranscriptionSegment)
                .where(TranscriptionSegment.session_id == sid)
                .where(TranscriptionSegment.original_speaker_label == label)
                .order_by(TranscriptionSegment.start_time)
            )
        )
        if rows:
            source_audio = rows[0].audio_file
            start = float(rows[0].start_time)
            end = float(rows[-1].end_time)

    models = getattr(cfg, "models", None)
    embedding_model = (
        str(getattr(models, "embedding_model", None) or "")
        or "nvidia/speakerverification_en_titanet_large"
    )

    try:
        result = gallery.enroll(
            sp,
            embedding,
            embedding_model=embedding_model,
            source_session_id=sid,
            source_audio_file=source_audio,
            start_time=start,
            end_time=end,
            force=bool(force),
        )
    except ValueError as exc:
        raise SessionsApiError(str(exc), status_code=422) from exc

    return {
        "enrollment_id": result.enrollment_id,
        "speaker_id": result.speaker_id,
        "display_name": result.display_name,
        "duration": result.duration,
        "quality_score": result.quality_score,
        "embedding_model": result.embedding_model,
        "embedding_dim": result.embedding_dim,
    }


def reidentify_session_api(cfg: Any, session_id: str) -> dict[str, Any]:
    from pawn_diarize.core.reidentify import reidentify_session
    from pawn_diarize.core.vault_transcript import push_session_transcript

    sid = (session_id or "").strip()
    if not sid:
        raise SessionsApiError("session_id is required", status_code=422)

    models = getattr(cfg, "models", None)
    try:
        result = reidentify_session(
            sid,
            cfg.db_dsn,
            embedding_model=getattr(models, "embedding_model", None),
            hf_token=getattr(models, "hf_token", None),
            speakers_config=getattr(cfg, "speakers", None),
        )
    except ValueError as exc:
        raise SessionsApiError(str(exc), status_code=422) from exc

    vault_refreshed = False
    vault_error = None
    if _transcript_path(cfg, sid):
        try:
            push_session_transcript(sid, db_dsn=cfg.db_dsn, cfg=cfg)
            vault_refreshed = True
        except Exception as exc:  # noqa: BLE001
            vault_error = str(exc)
            logger.info("vault refresh after reidentify skipped: %s", exc)

    return {
        "session_id": result.session_id,
        "segments_updated": result.segments_updated,
        "matched": dict(result.matched),
        "unmatched": list(result.unmatched),
        "summary": result.summary(),
        "vault_refreshed": vault_refreshed,
        "vault_error": vault_error,
    }


async def enqueue_session_reprocess(
    cfg: Any,
    session_id: str,
    *,
    confirm: bool,
    mode: str = "rediarize",
    backend: str = "nemo",
    language: Optional[str] = None,
    threshold: float = 0.7,
    cross_file_threshold: float = 0.55,
    margin: Optional[float] = None,
    job_id: Optional[str] = None,
) -> dict[str, Any]:
    """Validate session audio and enqueue ``rediarize`` / ``retranscribe`` on the diarize queue.

    Creates a ``session_reprocess`` vault_task (Jobs UI) and returns once the
    queue message is published. The diarize worker performs the ML work and
    refreshes the vault transcript.
    """
    import asyncio

    from pawn_diarize.core.reprocess import prepare_session_audio
    from pawn_server.core import jobs
    from pawn_server.core.diarize_publish import publish_to_diarize_queue
    from pawn_server.core.vault_tasks import new_task_id

    sid = (session_id or "").strip()
    if not sid:
        raise SessionsApiError("session_id is required", status_code=422)
    if not confirm:
        raise SessionsApiError("confirm must be true", status_code=422)

    mode_norm = (mode or "rediarize").strip().lower()
    if mode_norm not in ("rediarize", "retranscribe"):
        raise SessionsApiError("mode must be 'rediarize' or 'retranscribe'", status_code=422)
    backend_norm = (backend or "nemo").strip().lower()
    if backend_norm not in ("nemo", "whisper"):
        raise SessionsApiError("backend must be 'nemo' or 'whisper'", status_code=422)
    lang = (language or "").strip() or None
    if lang and backend_norm != "whisper":
        lang = None

    try:
        audio_files, _engine = await asyncio.to_thread(prepare_session_audio, sid, cfg.db_dsn, cfg)
    except ValueError as exc:
        raise SessionsApiError(str(exc), status_code=422) from exc

    queue_payload: dict[str, Any] = {
        "session": sid,
        "threshold": float(threshold),
        "cross_file_threshold": float(cross_file_threshold),
        "push_vault": True,
    }
    if mode_norm == "rediarize":
        if margin is not None:
            queue_payload["margin"] = float(margin)
        command = "rediarize"
    else:
        queue_payload["backend"] = backend_norm
        if lang:
            queue_payload["language"] = lang
        command = "retranscribe"

    effective_job_id = (job_id or "").strip() or new_task_id()
    queue_payload["job_id"] = effective_job_id
    title = (
        f"Rediarize session {sid}"
        if mode_norm == "rediarize"
        else (f"Retranscribe session {sid} ({backend_norm}" + (f", {lang}" if lang else "") + ")")
    )
    row = jobs.create_session_reprocess_job_row(
        cfg,
        job_id=effective_job_id,
        session_id=sid,
        mode=mode_norm,
        title=title,
        payload={
            "session": sid,
            "mode": mode_norm,
            "command": command,
            "audio_files": len(audio_files),
            "backend": backend_norm if mode_norm == "retranscribe" else None,
            "language": lang,
            "threshold": float(threshold),
            "cross_file_threshold": float(cross_file_threshold),
            "margin": float(margin) if margin is not None else None,
        },
    )
    _mark_processing(cfg, sid, mode=mode_norm, job_id=row.id)

    receipt = await publish_to_diarize_queue(
        cfg,
        command=command,
        payload=queue_payload,
    )

    if isinstance(receipt, str) and receipt.startswith("Error"):
        _mark_processing(cfg, sid, mode=mode_norm, job_id=row.id, clear=True)
        jobs.finish_session_reprocess_job(
            cfg,
            row.id,
            status="blocked",
            result=receipt[:2000],
            error_code="queue_publish_failed",
        )
        raise SessionsApiError(receipt, status_code=502)

    result_text = (
        f"Queued `{command}` for session `{sid}` "
        f"({len(audio_files)} audio file(s)). Transcript refreshes when the "
        f"diarize worker finishes."
    )
    # Stay running until the diarize worker clears the processing marker.
    jobs.finish_session_reprocess_job(
        cfg,
        row.id,
        status="running",
        result=result_text,
        payload_update={"receipt": receipt},
    )
    return {
        "job_id": row.id,
        "session_id": sid,
        "mode": mode_norm,
        "queued": True,
        "command": command,
        "audio_files": len(audio_files),
        "backend": backend_norm if mode_norm == "retranscribe" else None,
        "language": lang,
        "result": result_text,
    }


def ignore_session_speaker(
    cfg: Any, session_id: str, *, from_label: str, ignore: bool = True
) -> dict[str, Any]:
    """Mark a local label as ignored noise (or clear the ignore flag)."""
    from datetime import datetime, timezone

    sid = (session_id or "").strip()
    label = (from_label or "").strip()
    if not sid or not label:
        raise SessionsApiError("session_id and from_label are required", status_code=422)

    stats = _label_stats(cfg.db_dsn, [sid]).get(sid, {})
    if label not in stats and ignore:
        raise SessionsApiError(
            f"No spoken turns for label {label!r} in session {sid!r}",
            status_code=422,
        )

    gallery = SpeakerGallery(cfg.db_dsn, config=getattr(cfg, "speakers", None))
    if ignore:
        gallery.upsert_session_map(
            sid,
            label,
            speaker_id=None,
            display_name=None,
            match_score=None,
            match_method="ignore",
        )
    else:
        engine = get_engine(cfg.db_dsn)
        init_db(engine)
        with OrmSession(engine) as db:
            row = db.get(SessionSpeakerMap, (sid, label))
            if row is not None and str(row.match_method or "") == "ignore":
                db.delete(row)

    mapping = gallery.load_session_map(sid)
    trained = bool(_speakers_trained_flags(cfg.db_dsn, [sid]).get(sid))
    labels = _label_stats(cfg.db_dsn, [sid]).get(sid, {})
    speakers, unmapped, needs = _speakers_summary(labels, mapping, speakers_trained=trained)
    return {
        "session_id": sid,
        "from_label": label,
        "ignored": bool(ignore),
        "speakers": speakers,
        "unmapped_count": unmapped,
        "needs_speakers": needs,
        "speakers_trained": trained,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def set_speakers_trained(cfg: Any, session_id: str, *, trained: bool = True) -> dict[str, Any]:
    """Mark the session as speaker-trained (or clear the flag)."""
    from datetime import datetime, timezone

    sid = (session_id or "").strip()
    if not sid:
        raise SessionsApiError("session_id is required", status_code=422)

    engine = get_engine(cfg.db_dsn)
    init_db(engine)
    now = datetime.now(timezone.utc)
    with OrmSession(engine) as db:
        state = db.get(SessionState, sid)
        if state is None:
            if not trained:
                return {
                    "session_id": sid,
                    "speakers_trained": False,
                    "needs_speakers": True,
                }
            state = SessionState(
                session_id=sid,
                processed_files=[],
                speaker_embeddings={},
                time_cursor=0.0,
                speakers_trained_at=now,
                updated_at=now,
            )
            db.add(state)
        else:
            state.speakers_trained_at = now if trained else None
            state.updated_at = now

    labels = _label_stats(cfg.db_dsn, [sid]).get(sid, {})
    mapping = _session_maps(cfg.db_dsn, [sid]).get(sid, {})
    speakers, unmapped, needs = _speakers_summary(labels, mapping, speakers_trained=bool(trained))
    return {
        "session_id": sid,
        "speakers_trained": bool(trained),
        "speakers": speakers,
        "unmapped_count": unmapped,
        "needs_speakers": needs,
    }


def relabel_session_api(
    cfg: Any, session_id: str, *, from_label: str, to_label: str
) -> dict[str, Any]:
    from pawn_diarize.core.session_relabel import relabel_session_speaker
    from pawn_diarize.core.vault_transcript import refresh_transcript_after_relabel

    sid = (session_id or "").strip()
    src = (from_label or "").strip()
    dst = (to_label or "").strip()
    if not sid or not src or not dst:
        raise SessionsApiError("session_id, from, and to are required", status_code=422)
    try:
        result = relabel_session_speaker(cfg.db_dsn, sid, src, dst)
    except ValueError as exc:
        raise SessionsApiError(str(exc), status_code=422) from exc

    vault_refreshed = False
    vault_error = None
    try:
        refresh_transcript_after_relabel(sid, db_dsn=cfg.db_dsn, cfg=cfg)
        vault_refreshed = True
    except Exception as exc:  # noqa: BLE001
        vault_error = str(exc)
        logger.info("vault refresh after relabel skipped: %s", exc)

    return {
        "session_id": result.session_id,
        "from": result.from_label,
        "to": result.to_label,
        "segments_updated": result.segments_updated,
        "summary": result.summary(),
        "vault_refreshed": vault_refreshed,
        "vault_error": vault_error,
    }


async def extract_session(
    cfg: Any,
    session_id: str,
    *,
    items: bool = True,
    people: bool = False,
) -> dict[str, Any]:
    """Run coworker item extract and/or people refresh for one session.

    Items require ``coworker.enabled``. People refresh runs even when coworker
    is off (same as the CLI). Vault ``## Annotations`` are included in item
    extract via :func:`process_session`.
    """
    sid = (session_id or "").strip()
    if not sid:
        raise SessionsApiError("session_id is required", status_code=422)
    if not items and not people:
        raise SessionsApiError("Provide items=true and/or people=true", status_code=422)

    out: dict[str, Any] = {"session_id": sid, "items": None, "people": None}

    if items:
        if not getattr(getattr(cfg, "coworker", None), "enabled", False):
            raise SessionsApiError(
                "coworker is disabled; enable coworker to extract items "
                "(people-only extract still works)",
                status_code=400,
            )
        from pawn_agent.core.coworker.pipeline import process_session  # noqa: PLC0415

        try:
            out["items"] = await process_session(cfg, sid)
        except ValueError as exc:
            raise SessionsApiError(str(exc), status_code=422) from exc

    if people:
        from pawn_agent.core.people.refresh import refresh_people_for_session  # noqa: PLC0415

        out["people"] = await refresh_people_for_session(cfg, sid, force=True)

    return out


def _mark_processing(
    cfg: Any,
    session_id: str,
    *,
    mode: str,
    job_id: str,
    clear: bool = False,
) -> None:
    invalidate_session_files(session_id)
    engine = get_engine(cfg.db_dsn)
    if clear:
        set_session_processing(session_id, None, engine)
        return
    set_session_processing(
        session_id,
        {
            "mode": mode,
            "job_id": job_id,
            "since": datetime.now(timezone.utc).isoformat(),
        },
        engine,
    )


def _is_processing(db_dsn: str, session_id: str) -> bool:
    try:
        engine = get_engine(db_dsn)
        with OrmSession(engine) as db:
            row = db.get(SessionState, session_id)
            return bool(row is not None and row.processing)
    except Exception as exc:  # noqa: BLE001
        logger.debug("processing flag failed: %s", exc)
        return False


def _load_segments(db_dsn: str, session_id: str) -> list[Any]:
    engine = get_engine(db_dsn)
    with OrmSession(engine) as db:
        rows = list(
            db.scalars(
                select(TranscriptionSegment)
                .where(TranscriptionSegment.session_id == session_id)
                .order_by(TranscriptionSegment.start_time, TranscriptionSegment.segment_index)
            )
        )
        for row in rows:
            db.expunge(row)
        return rows


def select_samples(segments: list[dict[str, Any]], *, limit: int = 3) -> list[dict[str, Any]]:
    """The longest spoken segments, returned in time order."""
    spoken = [seg for seg in segments if str(seg.get("text") or "").strip()]
    spoken.sort(key=lambda seg: float(seg["end"]) - float(seg["start"]), reverse=True)
    picked = spoken[:limit]
    picked.sort(key=lambda seg: (float(seg["start"]), float(seg["end"])))
    return [
        {
            "start": float(seg["start"]),
            "end": float(seg["end"]),
            "text": str(seg["text"]).strip(),
        }
        for seg in picked
    ]


def _samples_by_label(rows: list[Any]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = str(row.original_speaker_label or "SPEAKER_00")
        grouped.setdefault(label, []).append(
            {
                "start": float(row.start_time or 0.0),
                "end": float(row.end_time or 0.0),
                "text": str(row.text or ""),
            }
        )
    return {label: select_samples(segs) for label, segs in grouped.items()}


def _suggest_speaker(gallery: Any, centroid: Any) -> tuple[Optional[str], Optional[float]]:
    """Best gallery hit for an unmapped centroid, even below the accept threshold."""
    try:
        import numpy as np

        hit = gallery.identify(np.asarray(centroid, dtype="float32"), threshold=0.0, margin=0.0)
    except Exception as exc:  # noqa: BLE001
        logger.debug("speaker suggestion failed: %s", exc)
        return None, None
    if not getattr(hit, "speaker_id", None):
        return None, None
    return str(hit.speaker_id), round(float(hit.score or 0.0), 4)


def _analysis_view(row: Any) -> Optional[dict[str, Any]]:
    if row is None:
        return None
    highlights = _parse_topics(getattr(row, "speaker_highlights", None))
    topics = _parse_topics(getattr(row, "key_topics", None))
    sentiment = str(getattr(row, "sentiment", "") or "").strip() or None
    return {
        "summary": getattr(row, "summary", None),
        "topics": topics,
        "tags": _parse_tags(getattr(row, "tags", None)),
        "sentiment": sentiment,
        "key_points": highlights or topics,
    }


def _analysis_path(cfg: Any, session_id: str) -> Optional[str]:
    vault = getattr(cfg, "vault", None)
    template = str(getattr(vault, "analysis_path_template", "") or "")
    root = str(getattr(vault, "agent_root", "") or "Pawn")
    if not template:
        return None
    try:
        path = template.format(agent_root=root, session_id=session_id)
    except (KeyError, IndexError, ValueError):
        return None
    try:
        from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

        store = vault_store_from_config(cfg)
        if store.exists(path):
            return path
    except Exception as exc:  # noqa: BLE001
        logger.debug("analysis path lookup failed: %s", exc)
    return None


def _side_chats(cfg: Any, session_id: str) -> list[dict[str, str]]:
    try:
        from pawn_server.core.portal_models import PortalChat  # noqa: PLC0415

        engine = get_engine(cfg.db_dsn)
        with OrmSession(engine) as db:
            rows = list(
                db.scalars(
                    select(PortalChat)
                    .where(PortalChat.origin.is_not(None))
                    .order_by(PortalChat.updated_at.desc())
                    .limit(200)
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("side chats lookup failed: %s", exc)
        return []
    out: list[dict[str, str]] = []
    for row in rows:
        origin = row.origin if isinstance(row.origin, dict) else {}
        if origin.get("kind") != "session" or str(origin.get("ref") or "") != session_id:
            continue
        out.append({"id": str(row.id), "title": str(row.title or "")})
    return out


def _word_list(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        return []
    out: list[dict[str, Any]] = []
    for word in raw:
        if not isinstance(word, dict):
            continue
        text = str(word.get("text") or word.get("word") or "")
        out.append(
            {
                "text": text,
                "start": float(word.get("start") or 0.0),
                "end": float(word.get("end") or 0.0),
            }
        )
    return out


def merge_turns(
    segments: list[dict[str, Any]], *, include_words: bool = False
) -> list[dict[str, Any]]:
    """Merge consecutive same-label segments into turns, in start order."""
    ordered = sorted(
        segments, key=lambda seg: (float(seg.get("start") or 0.0), str(seg.get("id") or ""))
    )
    turns: list[dict[str, Any]] = []
    for seg in ordered:
        label = str(seg.get("label") or "SPEAKER_00")
        piece: dict[str, Any] = {
            "id": str(seg.get("id") or ""),
            "start": float(seg.get("start") or 0.0),
            "end": float(seg.get("end") or 0.0),
            "text": str(seg.get("text") or ""),
        }
        if include_words:
            piece["words"] = list(seg.get("words") or [])
        if turns and turns[-1]["label"] == label:
            turns[-1]["end"] = max(float(turns[-1]["end"]), piece["end"])
            turns[-1]["segments"].append(piece)
            continue
        turns.append(
            {
                "id": piece["id"],
                "label": label,
                "start": piece["start"],
                "end": piece["end"],
                "segments": [piece],
            }
        )
    return turns


def audio_mime(path: str) -> Optional[str]:
    return _AUDIO_MIME.get(Path(path).suffix.lower())


def _skip_id3(head: bytes) -> int:
    """Offset past a leading ID3v2 tag (some FLAC/MP3 files carry one)."""
    if len(head) >= 10 and head[:3] == b"ID3":
        size = 0
        for byte in head[6:10]:
            size = (size << 7) | (byte & 0x7F)
        return 10 + size
    return 0


def sniff_audio_mime(head: bytes) -> Optional[str]:
    """Audio MIME type from the first bytes of a file, or None."""
    if not head:
        return None
    pos = _skip_id3(head)
    body = head[pos:]
    if body[:4] == b"fLaC":
        return "audio/flac"
    if pos and body[:2] and body[0] == 0xFF and (body[1] & 0xE0) == 0xE0:
        return "audio/mpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WAVE":
        return "audio/wav"
    if head[:4] == b"OggS":
        return "audio/ogg"
    if head[:4] == b"\x1a\x45\xdf\xa3":
        return "audio/webm"
    if head[4:8] == b"ftyp":
        return "audio/mp4"
    if pos or (len(head) > 1 and head[0] == 0xFF and (head[1] & 0xE0) == 0xE0):
        return "audio/mpeg"
    return None


def flac_duration(head: bytes) -> float:
    """Duration from the FLAC STREAMINFO block (always the first metadata block)."""
    pos = _skip_id3(head)
    if head[pos : pos + 4] != b"fLaC" or len(head) < pos + 8 + 18:
        return 0.0
    block = head[pos + 8 : pos + 8 + 34]
    if (head[pos + 4] & 0x7F) != 0 or len(block) < 18:
        return 0.0
    # Bytes 10..17: 20-bit sample rate, 3-bit channels, 5-bit bps, 36-bit total samples.
    packed = int.from_bytes(block[10:18], "big")
    rate = packed >> 44
    total = packed & ((1 << 36) - 1)
    if rate <= 0 or total <= 0:
        return 0.0
    return total / float(rate)


def wav_duration(head: bytes, size: int = 0) -> float:
    """Duration from RIFF ``fmt `` + ``data`` chunks. *size* covers streamed headers."""
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        return 0.0
    pos = 12
    byte_rate = 0
    while pos + 8 <= len(head):
        chunk_id = head[pos : pos + 4]
        (chunk_len,) = struct.unpack("<I", head[pos + 4 : pos + 8])
        data_at = pos + 8
        if chunk_id == b"fmt " and data_at + 12 <= len(head):
            (byte_rate,) = struct.unpack("<I", head[data_at + 8 : data_at + 12])
        elif chunk_id == b"data":
            if byte_rate <= 0:
                return 0.0
            if chunk_len in (0, 0xFFFFFFFF) or (size and data_at + chunk_len > size):
                chunk_len = max(0, size - data_at) if size else 0
            return chunk_len / float(byte_rate)
        pos = data_at + chunk_len + (chunk_len & 1)
    return 0.0


def duration_from_header(head: bytes, size: int = 0) -> float:
    """FLAC or WAV duration from header bytes; 0.0 when the format is not parsed."""
    return flac_duration(head) or wav_duration(head, size)


def _read_header(cfg: Any, path: str) -> tuple[bytes, int]:
    """First ``_HEADER_BYTES`` of a local or ``s3://`` file, plus its size."""
    if path.startswith("s3://"):
        from pawn_diarize.core.s3 import parse_s3_uri  # noqa: PLC0415

        client = _audio_client(cfg)
        bucket, key = parse_s3_uri(path, client.bucket)
        size = client.head_content_length(key, bucket)
        obj = client.get_object(key, bucket, range_header=f"bytes=0-{_HEADER_BYTES - 1}")
        body = obj["Body"]
        try:
            return body.read(), size
        finally:
            close = getattr(body, "close", None)
            if close is not None:
                close()
    file_path = Path(path)
    if not file_path.is_file():
        return b"", 0
    with file_path.open("rb") as handle:
        return handle.read(_HEADER_BYTES), file_path.stat().st_size


def probe_any_duration(cfg: Any, path: str) -> float:
    """Duration of a local or ``s3://`` file without downloading it all."""
    try:
        head, size = _read_header(cfg, path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("audio header read failed for %s: %s", path, exc)
        return 0.0
    duration = duration_from_header(head, size)
    if duration > 0:
        return duration
    if not path.startswith("s3://"):
        return probe_audio_duration(path)
    return 0.0


def audio_signature(token: str, session_id: str, index: int, exp: int) -> str:
    message = f"{session_id}|{index}|{exp}".encode()
    return hmac.new((token or "").encode(), message, hashlib.sha256).hexdigest()


def audio_signature_ok(
    token: str,
    session_id: str,
    index: int,
    exp: Any,
    sig: Optional[str],
    *,
    now: Optional[int] = None,
) -> bool:
    if not sig:
        return False
    try:
        exp_i = int(exp)
    except (TypeError, ValueError):
        return False
    current = int(time.time()) if now is None else int(now)
    if exp_i < current:
        return False
    expected = audio_signature(token, session_id, index, exp_i)
    return hmac.compare_digest(expected, str(sig))


def parse_byte_range(header: Optional[str], size: int) -> Optional[tuple[int, int]]:
    """Return an inclusive byte range, or None for the whole object.

    Raises ``SessionsApiError`` (416) when the range cannot be satisfied.
    """
    if header is None or not str(header).strip():
        return None
    raw = str(header).strip()
    if not raw.lower().startswith("bytes="):
        raise SessionsApiError(
            "Only bytes ranges are supported",
            status_code=416,
            headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
        )
    spec = raw.split("=", 1)[1].split(",")[0].strip()
    if "-" not in spec:
        raise SessionsApiError(
            "Range not satisfiable",
            status_code=416,
            headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
        )
    start_s, end_s = spec.split("-", 1)
    try:
        if start_s == "":
            length = int(end_s)
            if length <= 0 or size <= 0:
                raise ValueError
            start = max(0, size - length)
            end = size - 1
        else:
            start = int(start_s)
            if start < 0 or size <= 0 or start >= size:
                raise ValueError
            end = size - 1 if end_s == "" else min(int(end_s), size - 1)
            if end < start:
                raise ValueError
    except ValueError as exc:
        raise SessionsApiError(
            "Range not satisfiable",
            status_code=416,
            headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
        ) from exc
    return start, end


def resolve_file_offsets(
    files: list[str],
    stored: Optional[list[Any]],
    probe: Any,
    bounds: Optional[dict[str, tuple[float, float]]] = None,
) -> tuple[list[dict[str, Any]], bool]:
    """Map each file to ``{path, offset, duration}``.

    Returns ``(rows, probed)``. *probed* is true when a duration had to be
    measured because *stored* did not already cover every file. When a probe
    fails, *bounds* (``{path: (first segment start, last segment end)}``)
    estimates the duration; those rows carry ``estimated: True``.
    """
    known: dict[str, tuple[float, float]] = {}
    for item in stored or []:
        if not isinstance(item, dict):
            continue
        path = str(item.get("path") or "")
        if not path:
            continue
        if "start" in item and "end" in item:
            start = float(item.get("start") or 0.0)
            end = float(item.get("end") or start)
            known[path] = (start, max(0.0, end - start))
        elif "offset" in item and "duration" in item:
            known[path] = (
                float(item.get("offset") or 0.0),
                max(0.0, float(item.get("duration") or 0.0)),
            )
    if files and all(path in known for path in files):
        return (
            [
                {"path": path, "offset": known[path][0], "duration": known[path][1]}
                for path in files
            ],
            False,
        )
    offset = 0.0
    out: list[dict[str, Any]] = []
    for path in files:
        if path in known:
            duration = known[path][1]
        else:
            try:
                duration = max(0.0, float(probe(path) or 0.0))
            except Exception as exc:  # noqa: BLE001
                logger.debug("duration probe failed for %s: %s", path, exc)
                duration = 0.0
        out.append({"path": path, "offset": offset, "duration": duration})
        offset += duration
    if bounds and any(row["duration"] <= 0 for row in out):
        out = _estimate_offsets(out, bounds)
    return out, True


def _estimate_offsets(
    rows: list[dict[str, Any]],
    bounds: dict[str, tuple[float, float]],
) -> list[dict[str, Any]]:
    """Fill unknown durations from segment times (global seconds).

    A file ends where the next file's first segment starts; the last file ends
    at its last segment. Good enough for fixed-length recorder chunks.
    """
    out: list[dict[str, Any]] = []
    offset = 0.0
    for i, row in enumerate(rows):
        duration = float(row["duration"])
        estimated = False
        if duration <= 0:
            nxt = rows[i + 1]["path"] if i + 1 < len(rows) else None
            if nxt and nxt in bounds:
                duration = max(0.0, bounds[nxt][0] - offset)
            elif row["path"] in bounds:
                duration = max(0.0, bounds[row["path"]][1] - offset)
            estimated = True
        item = {"path": row["path"], "offset": offset, "duration": duration}
        if estimated:
            item["estimated"] = True
        out.append(item)
        offset += duration
    return out


def probe_audio_duration(path: str) -> float:
    """Duration in seconds for a local file. S3 objects return 0 (probed later)."""
    file_path = Path(path)
    if not file_path.is_file():
        return 0.0
    if file_path.suffix.lower() == ".wav":
        try:
            with wave.open(str(file_path), "rb") as handle:
                rate = handle.getframerate() or 1
                return handle.getnframes() / float(rate)
        except (wave.Error, EOFError, OSError):
            return 0.0
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0.0
    try:
        proc = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(file_path),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=20,
        )
    except (OSError, subprocess.TimeoutExpired):
        return 0.0
    try:
        return max(0.0, float((proc.stdout or "").strip() or 0.0))
    except ValueError:
        return 0.0


def _stored_offsets(db_dsn: str, session_id: str) -> list[Any]:
    try:
        engine = get_engine(db_dsn)
        with OrmSession(engine) as db:
            row = db.get(SessionState, session_id)
            raw = getattr(row, "file_offsets", None) if row is not None else None
            return list(raw or [])
    except Exception as exc:  # noqa: BLE001
        logger.debug("file_offsets load failed: %s", exc)
        return []


def _cache_offsets(db_dsn: str, session_id: str, rows: list[dict[str, Any]]) -> None:
    payload = [
        {
            "path": row["path"],
            "start": float(row["offset"]),
            "end": float(row["offset"]) + float(row["duration"]),
        }
        for row in rows
        if float(row.get("duration") or 0.0) > 0 and not row.get("estimated")
    ]
    if len(payload) != len(rows):
        return
    try:
        engine = get_engine(db_dsn)
        with OrmSession(engine) as db:
            state = db.get(SessionState, session_id)
            if state is None:
                return
            state.file_offsets = payload
            db.commit()
    except Exception as exc:  # noqa: BLE001
        logger.debug("file_offsets cache failed: %s", exc)


def _segment_bounds(db_dsn: str, session_id: str) -> dict[str, tuple[float, float]]:
    try:
        engine = get_engine(db_dsn)
        with OrmSession(engine) as db:
            rows = db.execute(
                select(
                    TranscriptionSegment.audio_file,
                    func.min(TranscriptionSegment.start_time),
                    func.max(TranscriptionSegment.end_time),
                )
                .where(TranscriptionSegment.session_id == session_id)
                .group_by(TranscriptionSegment.audio_file)
            ).all()
    except Exception as exc:  # noqa: BLE001
        logger.debug("segment bounds load failed: %s", exc)
        return {}
    return {str(r[0]): (float(r[1] or 0.0), float(r[2] or 0.0)) for r in rows if r[0]}


def invalidate_session_files(session_id: Optional[str] = None) -> None:
    """Drop cached ``session_files`` (one session, or all)."""
    with _FILES_LOCK:
        if session_id is None:
            _FILES_CACHE.clear()
        else:
            _FILES_CACHE.pop(session_id, None)


def _on_vault_event(event: dict[str, Any]) -> None:
    # A transcript push can follow new audio chunks; the cache is cheap to refill.
    if any("/Transcripts/" in str(p) for p in (event.get("paths") or [])):
        invalidate_session_files()


try:  # pragma: no cover - import side effect
    from pawn_server.core import vault_events as _vault_events  # noqa: PLC0415

    _vault_events.add_listener(_on_vault_event)
except Exception:  # noqa: BLE001
    pass


def session_files(cfg: Any, session_id: str, *, refresh: bool = False) -> list[dict[str, Any]]:
    """Ordered audio files with global offsets, mime and index (cached ~60 s)."""
    now = time.monotonic()
    if not refresh:
        with _FILES_LOCK:
            hit = _FILES_CACHE.get(session_id)
        if hit is not None and now - hit[0] < _FILES_TTL:
            return [dict(item) for item in hit[1]]
    files = _load_session_files(cfg, session_id)
    with _FILES_LOCK:
        _FILES_CACHE[session_id] = (now, files)
    return [dict(item) for item in files]


def _load_session_files(cfg: Any, session_id: str) -> list[dict[str, Any]]:
    from pawn_diarize.core.reprocess import ordered_session_audio_files  # noqa: PLC0415

    engine = get_engine(cfg.db_dsn)
    paths = ordered_session_audio_files(engine, session_id)
    stored = _stored_offsets(cfg.db_dsn, session_id)
    resolved, probed = resolve_file_offsets(
        paths,
        stored,
        lambda path: probe_any_duration(cfg, path),
        _segment_bounds(cfg.db_dsn, session_id),
    )
    if probed:
        _cache_offsets(cfg.db_dsn, session_id, resolved)
    files: list[dict[str, Any]] = []
    for index, row in enumerate(resolved):
        files.append(
            {
                "index": index,
                "audio_file": row["path"],
                "offset": float(row["offset"]),
                "duration": float(row["duration"]),
                "mime": audio_mime(row["path"]) or "",
                "estimated": bool(row.get("estimated")),
            }
        )
    return files


def sessions_missing_offsets(db_dsn: str, *, include_stored: bool = False) -> list[str]:
    """Session ids with segments whose ``file_offsets`` are not stored yet."""
    engine = get_engine(db_dsn)
    with OrmSession(engine) as db:
        ids = [
            str(r[0])
            for r in db.execute(
                select(TranscriptionSegment.session_id)
                .where(TranscriptionSegment.session_id.is_not(None))
                .group_by(TranscriptionSegment.session_id)
            ).all()
        ]
        if include_stored:
            return sorted(ids)
        stored = {
            str(r[0])
            for r in db.execute(
                select(SessionState.session_id).where(SessionState.file_offsets.is_not(None))
            ).all()
        }
    return sorted(sid for sid in ids if sid not in stored)


def backfill_file_offsets(cfg: Any, session_id: str, *, force: bool = False) -> dict[str, Any]:
    """Probe and store ``file_offsets`` for one session. Returns a small report."""
    from pawn_diarize.core.reprocess import ordered_session_audio_files  # noqa: PLC0415

    engine = get_engine(cfg.db_dsn)
    paths = ordered_session_audio_files(engine, session_id)
    stored = [] if force else _stored_offsets(cfg.db_dsn, session_id)
    rows, _probed = resolve_file_offsets(
        paths,
        stored,
        lambda path: probe_any_duration(cfg, path),
        _segment_bounds(cfg.db_dsn, session_id),
    )
    estimated = sum(1 for row in rows if row.get("estimated"))
    if rows and not estimated:
        _cache_offsets(cfg.db_dsn, session_id, rows)
    invalidate_session_files(session_id)
    return {
        "session_id": session_id,
        "files": len(rows),
        "estimated": estimated,
        "stored": bool(rows) and not estimated,
        "duration": sum(float(row["duration"]) for row in rows),
    }


def session_transcript(cfg: Any, session_id: str, *, words: bool = False) -> dict[str, Any]:
    sid = (session_id or "").strip()
    if not sid:
        raise SessionsApiError("session_id is required", status_code=422)
    rows = _load_segments(cfg.db_dsn, sid)
    if not rows:
        raise SessionsApiError(f"Unknown session: {sid}", status_code=404)
    try:
        gallery = SpeakerGallery(cfg.db_dsn, config=getattr(cfg, "speakers", None))
        mapping = gallery.load_session_map(sid)
    except Exception as exc:  # noqa: BLE001
        logger.debug("speaker map for transcript %s failed: %s", sid, exc)
        mapping = {}
    flat: list[dict[str, Any]] = []
    labels: list[str] = []
    for row in rows:
        label = str(row.original_speaker_label or "SPEAKER_00")
        if label not in labels:
            labels.append(label)
        piece: dict[str, Any] = {
            "id": str(row.id),
            "label": label,
            "start": float(row.start_time or 0.0),
            "end": float(row.end_time or 0.0),
            "text": str(row.text or ""),
        }
        if words:
            piece["words"] = _word_list(row.words)
        flat.append(piece)
    files = session_files(cfg, sid)
    duration = 0.0
    if flat:
        duration = max(float(seg["end"]) for seg in flat)
    if files:
        duration = max(duration, max(float(f["offset"]) + float(f["duration"]) for f in files))
    speakers = []
    for label in labels:
        mapped = mapping.get(label)
        speakers.append(
            {
                "label": label,
                "name": str(mapped.display_name) if mapped and mapped.display_name else None,
                "speaker_id": str(mapped.speaker_id) if mapped and mapped.speaker_id else None,
                "ignored": _is_ignored(mapped),
            }
        )
    return {
        "session_id": sid,
        "duration": duration,
        "files": files,
        "speakers": speakers,
        "turns": merge_turns(flat, include_words=words),
        "captures": _captures(cfg.db_dsn, sid),
    }


def _captures(db_dsn: str, session_id: str) -> list[dict[str, Any]]:
    try:
        from pawn_core.database import SessionCapture  # noqa: PLC0415

        engine = get_engine(db_dsn)
        with OrmSession(engine) as db:
            rows = list(
                db.scalars(
                    select(SessionCapture)
                    .where(SessionCapture.session_id == session_id)
                    .order_by(SessionCapture.audio_offset_s)
                )
            )
    except Exception as exc:  # noqa: BLE001
        logger.debug("captures load failed: %s", exc)
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        kind = str(row.kind or "")
        if kind not in ("screenshot", "annotation"):
            continue
        text = str(row.summary or row.text or "")
        out.append(
            {
                "kind": kind,
                "t": float(row.audio_offset_s or 0.0),
                "text": text,
                "image_key": row.vault_key,
            }
        )
    return out


def session_audio_urls(cfg: Any, session_id: str, base_url: str) -> dict[str, Any]:
    sid = (session_id or "").strip()
    if not sid:
        raise SessionsApiError("session_id is required", status_code=422)
    files = session_files(cfg, sid)
    if not files:
        raise SessionsApiError(f"No audio files for session {sid!r}", status_code=404)
    token = str(getattr(cfg, "api_token", "") or "")
    exp = int(time.time()) + _AUDIO_URL_TTL
    root = base_url.rstrip("/")
    listed = []
    for item in files:
        index = int(item["index"])
        sig = audio_signature(token, sid, index, exp)
        listed.append(
            {
                "index": index,
                "url": (
                    f"{root}/v1/sessions/{quote(sid, safe='')}/audio/{index}"
                    f"?sig={sig}&exp={exp}"
                ),
                "exp": exp,
                "mime": item["mime"],
                "offset": float(item["offset"]),
                "duration": float(item["duration"]),
                "estimated": bool(item.get("estimated")),
            }
        )
    return {"session_id": sid, "files": listed}


def _iter_slice(blob: bytes) -> Iterator[bytes]:
    step = 64 * 1024
    for start in range(0, len(blob), step):
        yield blob[start : start + step]


def open_session_audio(
    cfg: Any,
    session_id: str,
    index: int,
    range_header: Optional[str],
) -> tuple[int, str, dict[str, str], Iterator[bytes]]:
    """Open one session audio file, honouring an HTTP Range header."""
    sid = (session_id or "").strip()
    files = session_files(cfg, sid)
    match = next((item for item in files if int(item["index"]) == int(index)), None)
    if match is None:
        raise SessionsApiError("Audio file not found", status_code=404)
    path = str(match["audio_file"])
    mime = match["mime"] or None
    if not mime:
        try:
            head, _size = _read_header(cfg, path)
        except Exception as exc:  # noqa: BLE001
            logger.debug("audio sniff failed for %s: %s", path, exc)
            head = b""
        mime = sniff_audio_mime(head)
        if mime:
            with _FILES_LOCK:
                cached = _FILES_CACHE.get(sid)
                if cached is not None:
                    for item in cached[1]:
                        if int(item["index"]) == int(index):
                            item["mime"] = mime
    if not mime:
        raise SessionsApiError(
            f"Unsupported audio format: {Path(path).name}",
            status_code=415,
        )
    if path.startswith("s3://"):
        return _open_s3_audio(cfg, path, mime, range_header)
    return _open_local_audio(path, mime, range_header)


def _open_local_audio(
    path: str,
    mime: str,
    range_header: Optional[str],
) -> tuple[int, str, dict[str, str], Iterator[bytes]]:
    file_path = Path(path)
    if not file_path.is_file():
        raise SessionsApiError("Audio file not found", status_code=404)
    size = file_path.stat().st_size
    span = parse_byte_range(range_header, size)
    if span is None:
        start, end = 0, max(0, size - 1)
        status = 200
        length = size
    else:
        start, end = span
        status = 206
        length = end - start + 1
    blob = file_path.read_bytes()[start : start + length] if size else b""
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(length),
    }
    if status == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return status, mime, headers, _iter_slice(blob)


def _audio_client(cfg: Any) -> Any:
    s3 = getattr(cfg, "s3", None)
    if s3 is None:
        raise SessionsApiError("S3 is not configured", status_code=422)
    data = s3.model_dump() if hasattr(s3, "model_dump") else dict(s3)
    from pawn_diarize.core.s3 import S3Client  # noqa: PLC0415

    return S3Client.from_dict(data)


def _open_s3_audio(
    cfg: Any,
    uri: str,
    mime: str,
    range_header: Optional[str],
) -> tuple[int, str, dict[str, str], Iterator[bytes]]:
    from pawn_diarize.core.s3 import parse_s3_uri  # noqa: PLC0415

    client = _audio_client(cfg)
    bucket, key = parse_s3_uri(uri, client.bucket)
    try:
        size = client.head_content_length(key, bucket)
    except Exception as exc:  # noqa: BLE001
        raise SessionsApiError(f"Audio object not found: {exc}", status_code=404) from exc
    span = parse_byte_range(range_header, size)
    header = None
    status = 200
    if span is not None:
        header = f"bytes={span[0]}-{span[1]}"
        status = 206
    try:
        obj = client.get_object(key, bucket, range_header=header)
    except Exception as exc:  # noqa: BLE001
        if "InvalidRange" in str(exc):
            raise SessionsApiError(
                "Range not satisfiable",
                status_code=416,
                headers={"Content-Range": f"bytes */{size}", "Accept-Ranges": "bytes"},
            ) from exc
        raise SessionsApiError(f"Audio read failed: {exc}", status_code=502) from exc
    body = obj["Body"]
    length = int(obj.get("ContentLength") or (0 if span is None else span[1] - span[0] + 1))
    headers = {"Accept-Ranges": "bytes", "Content-Length": str(length)}
    content_range = obj.get("ContentRange")
    if status == 206:
        headers["Content-Range"] = str(content_range or f"bytes {span[0]}-{span[1]}/{size}")

    def chunks() -> Iterator[bytes]:
        try:
            reader = getattr(body, "iter_chunks", None)
            if reader is not None:
                yield from reader(64 * 1024)
            else:
                while True:
                    block = body.read(64 * 1024)
                    if not block:
                        break
                    yield block
        finally:
            close = getattr(body, "close", None)
            if close is not None:
                close()

    return status, mime, headers, chunks()


def _bulk_session_ids(
    cfg: Any,
    *,
    session_ids: Optional[list[str]],
    needs_speakers: bool,
    limit: int,
) -> list[str]:
    cap = max(1, min(int(limit or 50), 50))
    if session_ids:
        out: list[str] = []
        for raw in session_ids:
            sid = str(raw or "").strip()
            if sid and sid not in out:
                out.append(sid)
            if len(out) >= cap:
                break
        if not out:
            raise SessionsApiError("session_ids is empty", status_code=422)
        return out
    if needs_speakers:
        page = list_sessions_page(cfg, limit=cap, needs_speakers=True)
        return [str(row["session_id"]) for row in page["data"]]
    raise SessionsApiError("Provide session_ids or needs_speakers", status_code=422)


async def run_bulk_reidentify(cfg: Any, job_id: str, session_ids: list[str]) -> None:
    """Reidentify each session, then mark *job_id* done or blocked."""
    from pawn_server.core import jobs  # noqa: PLC0415

    errors: list[str] = []
    for sid in session_ids:
        try:
            import asyncio

            await asyncio.to_thread(reidentify_session_api, cfg, sid)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{sid}: {exc}")
            logger.info("bulk reidentify %s failed: %s", sid, exc)
    if errors:
        jobs.finish_session_reprocess_job(
            cfg,
            job_id,
            status="blocked",
            result="; ".join(errors)[:2000],
            error_code="reidentify_failed",
        )
        return
    jobs.finish_session_reprocess_job(
        cfg,
        job_id,
        status="done",
        result=f"Reidentified {len(session_ids)} session(s).",
    )


def queue_bulk_reidentify(
    cfg: Any,
    *,
    session_ids: Optional[list[str]] = None,
    needs_speakers: bool = False,
    limit: int = 50,
) -> dict[str, Any]:
    """Create a job row and return the ids the background task should run."""
    ids = _bulk_session_ids(
        cfg, session_ids=session_ids, needs_speakers=needs_speakers, limit=limit
    )
    from pawn_server.core import jobs  # noqa: PLC0415
    from pawn_server.core.vault_tasks import new_task_id  # noqa: PLC0415

    job_id = new_task_id()
    title = f"Reidentify {len(ids)} session(s)"
    jobs.create_session_reprocess_job_row(
        cfg,
        job_id=job_id,
        session_id=ids[0],
        mode="reidentify",
        title=title,
        payload={"session_ids": ids, "mode": "reidentify"},
    )
    return {"job_id": job_id, "session_ids": ids, "queued": True}
