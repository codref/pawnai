"""HTTP helpers for Sessions management (list, detail, speaker train).

Wraps Speakers gallery + reidentify/relabel without loading heavy ML for
list/detail. Session enroll uses ``session_state`` centroids (no audio).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

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
from pawn_diarize.core.database import init_db, load_session_state
from pawn_diarize.core.speaker_gallery import SpeakerGallery

logger = logging.getLogger(__name__)

_ANON_SPEAKER_RE = re.compile(r"^SPEAKER_\d+$")
_SAMPLE_CHARS = 160
_TOPIC_SPLIT_RE = re.compile(r"[\n,;•]+")


class SessionsApiError(Exception):
    """Domain error with an HTTP status code for the API layer."""

    def __init__(self, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


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
    """session_id → local_label → {duration_s, segment_count}."""
    if not session_ids:
        return {}
    engine = get_engine(db_dsn)
    out: dict[str, dict[str, dict[str, Any]]] = {}
    try:
        with OrmSession(engine) as db:
            rows = db.execute(
                select(
                    TranscriptionSegment.session_id,
                    TranscriptionSegment.original_speaker_label,
                    func.min(TranscriptionSegment.start_time).label("first_start"),
                    func.max(TranscriptionSegment.end_time).label("last_end"),
                    func.count().label("n"),
                )
                .where(TranscriptionSegment.session_id.in_(session_ids))
                .group_by(
                    TranscriptionSegment.session_id,
                    TranscriptionSegment.original_speaker_label,
                )
            ).all()
            for row in rows:
                sid = str(row.session_id)
                label = str(row.original_speaker_label or "SPEAKER_00")
                duration = float(row.last_end or 0.0) - float(row.first_start or 0.0)
                out.setdefault(sid, {})[label] = {
                    "duration_s": max(0.0, duration),
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


def _speakers_summary(
    labels: dict[str, dict[str, Any]],
    mapping: dict[str, SessionSpeakerMap],
) -> tuple[list[str], int, bool]:
    """Return (speaker pills, unmapped_count, needs_speakers)."""
    pills: list[str] = []
    unmapped = 0
    for label in sorted(labels.keys()):
        mapped = mapping.get(label)
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
    # Needs training when any anonymous label remains.
    needs = any(_is_anon(label) for label in labels) or unmapped > 0
    # Prefer counting anonymous labels as the attention signal.
    anon_count = sum(1 for label in labels if _is_anon(label))
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


def list_sessions_page(
    cfg: Any,
    *,
    limit: int = 50,
    q: str = "",
    needs_speakers: bool = False,
    speaker: str = "",
) -> dict[str, Any]:
    """Recent sessions plus speaker facets for the Sessions tab."""
    fetch_limit = min(300, max(limit * 4, 120))
    # Broad fetch; filter q/speaker in Python so search hits title/tags/speakers.
    candidates = list_session_candidates_impl(cfg, name_filter="", limit=fetch_limit)
    session_ids = [c.session_id for c in candidates]
    label_stats = _label_stats(cfg.db_dsn, session_ids)
    maps = _session_maps(cfg.db_dsn, session_ids)
    analyses = _load_analyses(cfg.db_dsn, session_ids)

    enriched: list[dict[str, Any]] = []
    speaker_counts: dict[str, int] = {}
    needs_total = 0
    for cand in candidates:
        labels = label_stats.get(cand.session_id, {})
        mapping = maps.get(cand.session_id, {})
        speakers, unmapped_count, needs = _speakers_summary(labels, mapping)
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
            "transcript_path": _transcript_path(cfg, cand.session_id),
            "speakers": speakers,
            "unmapped_count": unmapped_count,
            "needs_speakers": needs,
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

    facets_speakers = [
        {"name": name, "count": count}
        for name, count in sorted(speaker_counts.items(), key=lambda kv: (-kv[1], kv[0].lower()))[
            :12
        ]
    ]

    speaker_filter = (speaker or "").strip().lower()
    out: list[dict[str, Any]] = []
    for row in enriched:
        if needs_speakers and not row["needs_speakers"]:
            continue
        if speaker_filter:
            names = [str(s).lower() for s in (row.get("speakers") or [])]
            if speaker_filter not in names:
                continue
        if q.strip() and not _row_matches_query(row, q):
            continue
        out.append(row)
        if len(out) >= limit:
            break

    return {
        "data": out,
        "facets": {
            "speakers": facets_speakers,
            "needs_speakers": needs_total,
        },
    }


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

    labels_out: list[dict[str, Any]] = []
    for label in sorted(labels_map.keys()):
        info = labels_map[label]
        mapped = mapping.get(label)
        centroid = None
        if isinstance(prior, dict):
            entry = prior.get(label)
            if isinstance(entry, dict):
                centroid = entry.get("embedding")
        labels_out.append(
            {
                "local_label": label,
                "duration_s": float(info.get("duration_s") or 0.0),
                "sample_text": _sample_text(cfg.db_dsn, sid, label),
                "mapped_speaker_id": (
                    str(mapped.speaker_id) if mapped and mapped.speaker_id else None
                ),
                "mapped_name": (
                    str(mapped.display_name) if mapped and mapped.display_name else None
                ),
                "has_centroid": centroid is not None,
            }
        )

    speakers_summary, unmapped_count, needs = _speakers_summary(labels_map, mapping)
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
        "labels": labels_out,
        "gallery": gallery_out,
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
        raise SessionsApiError(
            "mode must be 'rediarize' or 'retranscribe'", status_code=422
        )
    backend_norm = (backend or "nemo").strip().lower()
    if backend_norm not in ("nemo", "whisper"):
        raise SessionsApiError("backend must be 'nemo' or 'whisper'", status_code=422)
    lang = (language or "").strip() or None
    if lang and backend_norm != "whisper":
        lang = None

    try:
        audio_files, _engine = await asyncio.to_thread(
            prepare_session_audio, sid, cfg.db_dsn, cfg
        )
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
    title = (
        f"Rediarize session {sid}"
        if mode_norm == "rediarize"
        else (
            f"Retranscribe session {sid} ({backend_norm}"
            + (f", {lang}" if lang else "")
            + ")"
        )
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

    receipt = await publish_to_diarize_queue(
        cfg,
        command=command,
        payload=queue_payload,
    )

    if isinstance(receipt, str) and receipt.startswith("Error"):
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
    jobs.finish_session_reprocess_job(
        cfg,
        row.id,
        status="done",
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
