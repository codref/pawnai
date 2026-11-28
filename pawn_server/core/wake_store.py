"""Shared wake-word training store: labelled clips in, published models out.

Assistants (phone and laptop) upload the 2 s clip that triggered each wake,
labelled by what happened next. The Linux trainer (``pawn-assistant
wake-train pull``) downloads them into its project inbox, retrains, and
publishes the model back (``wake-train publish``); every assistant then
fetches the active model. Files live on disk under ``devices.wake_dir``
(default ``{state_dir}/wake``); rows in ``wake_clips`` / ``wake_models``.

See docs/ASSISTANTS.md.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import func, select

from pawn_agent.utils.db import WakeClip, WakeModel, _get_session

logger = logging.getLogger(__name__)

#: Same labels as pawn-assistant's harvest inbox.
CLIP_LABELS: tuple[str, ...] = ("confirmed", "barge", "false_wake", "unsure", "near_miss")
_VERSION_RE = re.compile(r"^[A-Za-z0-9._-]{1,64}$")


class WakeStoreError(Exception):
    def __init__(self, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def wake_dir(cfg: Any) -> Path:
    raw = str(getattr(getattr(cfg, "devices", None), "wake_dir", "") or "")
    if raw:
        return Path(raw).expanduser()
    from pawn_agent.utils.config import state_dir_of  # noqa: PLC0415

    return state_dir_of(cfg) / "wake"


def _limit(cfg: Any, name: str, default: int) -> int:
    return int(getattr(getattr(cfg, "devices", None), name, default) or default)


def clip_view(row: WakeClip) -> dict[str, Any]:
    return {
        "id": row.id,
        "device_id": row.device_id,
        "label": row.label,
        "score": row.score,
        "size": row.size,
        "meta": dict(row.meta or {}),
        "captured_at": _iso(row.captured_at),
        "created_at": _iso(row.created_at),
    }


def model_view(row: WakeModel) -> dict[str, Any]:
    return {
        "version": row.version,
        "name": row.name,
        "sha256": row.sha256,
        "size": row.size,
        "notes": row.notes,
        "thresholds": dict(row.thresholds or {}),
        "active": bool(row.active),
        "created_at": _iso(row.created_at),
        "url": f"/v1/assistants/wake/models/{row.version}",
    }


# ── Clips ─────────────────────────────────────────────────────────────────────


def _check_wav(data: bytes) -> None:
    if len(data) < 44 or data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise WakeStoreError("clip must be a WAV file")


def add_clip(
    cfg: Any,
    data: bytes,
    *,
    label: str,
    device_id: Optional[str] = None,
    score: Optional[float] = None,
    captured_at: Optional[datetime] = None,
    meta: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    label = (label or "").strip().lower()
    if label not in CLIP_LABELS:
        raise WakeStoreError(f"label must be one of {', '.join(CLIP_LABELS)}")
    max_bytes = _limit(cfg, "wake_clip_max_bytes", 1_000_000)
    if len(data) > max_bytes:
        raise WakeStoreError(f"clip is larger than {max_bytes} bytes", status_code=413)
    _check_wav(data)
    clip_id = uuid.uuid4().hex
    now = _now()
    rel = Path("clips") / label / f"{(captured_at or now):%Y%m%d-%H%M%S}_{clip_id[:8]}.wav"
    path = wake_dir(cfg) / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    clean_meta = {k: v for k, v in (meta or {}).items() if isinstance(k, str)}
    if len(json.dumps(clean_meta, default=str)) > 4000:
        clean_meta = {}
    with _get_session(cfg.db_dsn) as db:
        row = WakeClip(
            id=clip_id,
            device_id=device_id,
            label=label,
            score=score,
            path=str(rel),
            size=len(data),
            meta=clean_meta,
            captured_at=captured_at,
            created_at=now,
        )
        db.add(row)
        db.flush()
        return clip_view(row)


def list_clips(
    dsn: str,
    *,
    since: Optional[datetime] = None,
    label: Optional[str] = None,
    device_id: Optional[str] = None,
    limit: int = 500,
) -> list[dict[str, Any]]:
    with _get_session(dsn) as db:
        stmt = select(WakeClip).order_by(WakeClip.created_at).limit(max(1, min(limit, 5000)))
        if since is not None:
            stmt = stmt.where(WakeClip.created_at > since)
        if label:
            stmt = stmt.where(WakeClip.label == label)
        if device_id:
            stmt = stmt.where(WakeClip.device_id == device_id)
        return [clip_view(row) for row in db.scalars(stmt)]


def clip_counts(dsn: str, *, device_id: Optional[str] = None) -> dict[str, int]:
    with _get_session(dsn) as db:
        stmt = select(WakeClip.label, func.count()).group_by(WakeClip.label)
        if device_id:
            stmt = stmt.where(WakeClip.device_id == device_id)
        return {str(label): int(n) for label, n in db.execute(stmt)}


def clip_path(cfg: Any, clip_id: str) -> Path:
    with _get_session(cfg.db_dsn) as db:
        row = db.get(WakeClip, clip_id)
        if row is None:
            raise WakeStoreError("Unknown clip", status_code=404)
        rel = row.path
    path = wake_dir(cfg) / rel
    if not path.is_file():
        raise WakeStoreError("Clip file is missing", status_code=404)
    return path


def delete_clip(cfg: Any, clip_id: str) -> None:
    with _get_session(cfg.db_dsn) as db:
        row = db.get(WakeClip, clip_id)
        if row is None:
            raise WakeStoreError("Unknown clip", status_code=404)
        rel = row.path
        db.delete(row)
    (wake_dir(cfg) / rel).unlink(missing_ok=True)


def purge_clips(cfg: Any, *, older_than_days: Optional[int] = None) -> int:
    days = older_than_days if older_than_days is not None else _limit(cfg, "wake_clip_retention_days", 90)
    if days <= 0:
        return 0
    cutoff = _now() - timedelta(days=days)
    with _get_session(cfg.db_dsn) as db:
        rows = db.scalars(select(WakeClip).where(WakeClip.created_at < cutoff)).all()
        paths = [row.path for row in rows]
        for row in rows:
            db.delete(row)
    for rel in paths:
        (wake_dir(cfg) / rel).unlink(missing_ok=True)
    return len(paths)


# ── Models ────────────────────────────────────────────────────────────────────


def publish_model(
    cfg: Any,
    data: bytes,
    *,
    version: Optional[str] = None,
    name: str = "hey_pawn",
    notes: Optional[str] = None,
    thresholds: Optional[dict[str, Any]] = None,
    activate: bool = True,
) -> dict[str, Any]:
    max_bytes = _limit(cfg, "wake_model_max_bytes", 20_000_000)
    if not data:
        raise WakeStoreError("model file is empty")
    if len(data) > max_bytes:
        raise WakeStoreError(f"model is larger than {max_bytes} bytes", status_code=413)
    now = _now()
    version = (version or "").strip() or now.strftime("v%Y%m%d-%H%M%S")
    if not _VERSION_RE.match(version):
        raise WakeStoreError("version may only use letters, digits, '.', '_' and '-'")
    name = re.sub(r"[^A-Za-z0-9_-]", "_", (name or "hey_pawn").strip())[:40] or "hey_pawn"
    digest = hashlib.sha256(data).hexdigest()
    rel = Path("models") / f"{name}.{version}.onnx"
    path = wake_dir(cfg) / rel
    with _get_session(cfg.db_dsn) as db:
        if db.get(WakeModel, version) is not None:
            raise WakeStoreError(f"Version {version} already exists", status_code=409)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        if activate:
            for other in db.scalars(select(WakeModel).where(WakeModel.active.is_(True))):
                other.active = False
        row = WakeModel(
            version=version,
            name=name,
            path=str(rel),
            sha256=digest,
            size=len(data),
            notes=(notes or "").strip()[:2000] or None,
            thresholds={k: float(v) for k, v in (thresholds or {}).items() if isinstance(v, (int, float))},
            active=activate,
            created_at=now,
        )
        db.add(row)
        db.flush()
        return model_view(row)


def activate_model(dsn: str, version: str) -> dict[str, Any]:
    with _get_session(dsn) as db:
        row = db.get(WakeModel, version)
        if row is None:
            raise WakeStoreError("Unknown model version", status_code=404)
        for other in db.scalars(select(WakeModel).where(WakeModel.active.is_(True))):
            other.active = False
        row.active = True
        return model_view(row)


def current_model(dsn: str) -> Optional[dict[str, Any]]:
    with _get_session(dsn) as db:
        row = db.scalar(
            select(WakeModel).where(WakeModel.active.is_(True)).order_by(WakeModel.created_at.desc())
        )
        return model_view(row) if row is not None else None


def list_models(dsn: str) -> list[dict[str, Any]]:
    with _get_session(dsn) as db:
        return [model_view(row) for row in db.scalars(select(WakeModel).order_by(WakeModel.created_at.desc()))]


def model_path(cfg: Any, version: str) -> Path:
    with _get_session(cfg.db_dsn) as db:
        row = db.get(WakeModel, version)
        if row is None:
            raise WakeStoreError("Unknown model version", status_code=404)
        rel = row.path
    path = wake_dir(cfg) / rel
    if not path.is_file():
        raise WakeStoreError("Model file is missing", status_code=404)
    return path
