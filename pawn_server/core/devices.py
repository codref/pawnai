"""Phone bridge domain logic: device registry, auth, event ingest, purge.

Owns: ``devices`` / ``phone_events`` rows and the ``phone:<device>:<event>``
knowledge chunks derived from them.
Does not own: HTTP routing (devices_api.py) or the agent-facing read tools
(pawn_agent/tools/phone_impl.py).

All functions are synchronous; the API layer drives them with
``asyncio.to_thread``. See docs/PHONE_BRIDGE.md.
"""

from __future__ import annotations

import hashlib
import hmac
import logging
import secrets
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Optional

from sqlalchemy import delete, select

from pawn_agent.utils.db import Device, DeviceCommand, KnowledgeChunk, PhoneEvent, _get_session

logger = logging.getLogger(__name__)

EVENT_KINDS: tuple[str, ...] = ("call", "sms", "notification", "location", "status")
_MAX_TEXT = 4000
_MAX_KEY = 200


class DeviceError(Exception):
    """Domain error with an HTTP status code for the API layer."""

    def __init__(self, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


@dataclass
class IngestResult:
    accepted: int
    duplicates: int
    rejected: int
    to_index: list[tuple[str, str, str]]  # (source_kind, source_ref, text)
    # New events eligible for the coworker pipeline (source_kind, source_ref, text).
    to_coworker: list[tuple[str, str, str]] = field(default_factory=list)


def _now() -> datetime:
    """Naive UTC, matching the other agent tables."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def source_ref(device_id: str, event_id: str) -> str:
    return f"phone:{device_id}:{event_id}"


def _clean_categories(raw: Optional[Iterable[Any]]) -> list[str]:
    out: list[str] = []
    for item in raw or []:
        kind = str(item or "").strip().lower()
        if kind in EVENT_KINDS and kind not in out:
            out.append(kind)
    return out


def _clean_capabilities(raw: Optional[Iterable[Any]]) -> list[str]:
    from pawn_server.core.device_commands import TOOLS  # noqa: PLC0415

    out: list[str] = []
    for item in raw or []:
        name = str(item or "").strip().lower()
        if name in TOOLS and name not in out:
            out.append(name)
    return out


def device_view(row: Device) -> dict[str, Any]:
    return {
        "device_id": row.id,
        "name": row.name,
        "enabled_categories": list(row.enabled_categories or []),
        "capabilities": list(row.capabilities or []),
        "unifiedpush": bool(row.unifiedpush_endpoint),
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "last_seen": row.last_seen.isoformat() if row.last_seen else None,
    }


# ── Registry / auth ───────────────────────────────────────────────────────────


def register_device(dsn: str, name: str) -> dict[str, Any]:
    """Create a device and return its id plus the one-time plaintext token."""
    label = (name or "").strip()[:80] or "phone"
    device_id = uuid.uuid4().hex
    token = secrets.token_urlsafe(32)
    now = _now()
    with _get_session(dsn) as db:
        db.add(
            Device(
                id=device_id,
                name=label,
                token_hash=hash_token(token),
                enabled_categories=[],
                created_at=now,
                last_seen=now,
            )
        )
    return {"device_id": device_id, "device_token": token, "name": label}


def authenticate(dsn: str, device_id: str, token: Optional[str]) -> bool:
    """True when *token* is this device's token (constant-time compare)."""
    if not token:
        return False
    with _get_session(dsn) as db:
        row = db.get(Device, device_id)
        if row is None:
            return False
        return hmac.compare_digest(row.token_hash, hash_token(token))


def get_device(dsn: str, device_id: str) -> dict[str, Any]:
    with _get_session(dsn) as db:
        row = db.get(Device, device_id)
        if row is None:
            raise DeviceError("Unknown device", status_code=404)
        return device_view(row)


def list_devices(dsn: str) -> list[dict[str, Any]]:
    with _get_session(dsn) as db:
        rows = db.scalars(select(Device).order_by(Device.created_at)).all()
        return [device_view(row) for row in rows]


def update_device(
    dsn: str,
    device_id: str,
    *,
    name: Optional[str] = None,
    unifiedpush_endpoint: Optional[str] = None,
    enabled_categories: Optional[list[str]] = None,
    capabilities: Optional[list[str]] = None,
) -> tuple[dict[str, Any], list[str]]:
    """Update a device. Returns ``(view, newly_disabled_categories)``.

    Turning a category off drops what the server already holds for it, so the
    phone toggle means "the server no longer knows this", not just "stop".
    """
    with _get_session(dsn) as db:
        row = db.get(Device, device_id)
        if row is None:
            raise DeviceError("Unknown device", status_code=404)
        if name is not None and name.strip():
            row.name = name.strip()[:80]
        if unifiedpush_endpoint is not None:
            endpoint = unifiedpush_endpoint.strip()
            if endpoint and not endpoint.startswith(("https://", "http://")):
                raise DeviceError("unifiedpush_endpoint must be an http(s) URL")
            row.unifiedpush_endpoint = endpoint or None
        disabled: list[str] = []
        if enabled_categories is not None:
            new = _clean_categories(enabled_categories)
            disabled = [k for k in (row.enabled_categories or []) if k not in new]
            row.enabled_categories = new
        if capabilities is not None:
            row.capabilities = _clean_capabilities(capabilities)
        row.last_seen = _now()
        view = device_view(row)
    for kind in disabled:
        purge_events(dsn, device_id, kind=kind)
    return view, disabled


def delete_device(dsn: str, device_id: str) -> int:
    """Delete a device and everything derived from it. Returns events removed."""
    removed = purge_events(dsn, device_id)
    with _get_session(dsn) as db:
        db.execute(delete(DeviceCommand).where(DeviceCommand.device_id == device_id))
        row = db.get(Device, device_id)
        if row is None:
            raise DeviceError("Unknown device", status_code=404)
        db.delete(row)
    return removed


def purge_events(
    dsn: str,
    device_id: str,
    *,
    kind: Optional[str] = None,
    older_than: Optional[datetime] = None,
) -> int:
    """Delete phone events (and their knowledge chunks). Returns rows removed."""
    with _get_session(dsn) as db:
        stmt = select(PhoneEvent.id).where(PhoneEvent.device_id == device_id)
        if kind:
            stmt = stmt.where(PhoneEvent.kind == kind)
        if older_than is not None:
            stmt = stmt.where(PhoneEvent.ts < older_than)
        ids = list(db.scalars(stmt).all())
        if not ids:
            return 0
        refs = [source_ref(device_id, event_id) for event_id in ids]
        for start in range(0, len(ids), 500):
            db.execute(delete(PhoneEvent).where(PhoneEvent.id.in_(ids[start : start + 500])))
            db.execute(
                delete(KnowledgeChunk).where(KnowledgeChunk.source_ref.in_(refs[start : start + 500]))
            )
        return len(ids)


# ── Ingest ────────────────────────────────────────────────────────────────────


def parse_ts(raw: Any) -> datetime:
    """Accept epoch millis / seconds or ISO-8601; return naive UTC."""
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        value = float(raw)
        if value > 1e11:  # millis
            value /= 1000.0
        return datetime.fromtimestamp(value, tz=timezone.utc).replace(tzinfo=None)
    if isinstance(raw, str) and raw.strip():
        text = raw.strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is not None:
            parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
        return parsed
    raise ValueError("missing ts")


def event_text(kind: str, payload: dict[str, Any], text: Optional[str]) -> str:
    """Searchable one-liner for an event (used for knowledge indexing)."""
    if text and text.strip():
        return text.strip()[:_MAX_TEXT]
    who = payload.get("name") or payload.get("address") or payload.get("number") or ""
    if kind == "sms":
        direction = payload.get("direction") or "received"
        return f"SMS {direction} {who}: {payload.get('body') or ''}".strip()[:_MAX_TEXT]
    if kind == "call":
        direction = payload.get("direction") or "call"
        duration = payload.get("duration_s")
        tail = f" ({duration}s)" if duration else ""
        return f"Phone call {direction} {who}{tail}".strip()
    if kind == "notification":
        app = payload.get("app") or payload.get("package") or ""
        parts = [p for p in (payload.get("title"), payload.get("text")) if p]
        return f"Notification {app}: {' — '.join(str(p) for p in parts)}".strip()[:_MAX_TEXT]
    return ""


# Directions that can carry a follow-up for the user (see coworker_eligible).
_COWORKER_DIRECTIONS: dict[str, str] = {"sms": "received", "call": "missed"}


def coworker_eligible(kind: str, payload: dict[str, Any], kinds: Iterable[str]) -> bool:
    """True for a received SMS or a missed call whose kind is in *kinds*."""
    wanted = _COWORKER_DIRECTIONS.get(kind)
    if wanted is None or kind not in set(kinds):
        return False
    return str(payload.get("direction") or "").strip().lower() == wanted


def ingest_events(
    dsn: str,
    device_id: str,
    events: list[dict[str, Any]],
    *,
    index_kinds: Iterable[str] = (),
    retention_days: int = 0,
    coworker_kinds: Iterable[str] = (),
) -> IngestResult:
    """Store events idempotently on ``(device_id, dedup_key)``.

    Events of a kind the device has not enabled are rejected — the server
    keeps its own copy of the phone's toggles so a stale outbox cannot leak a
    category the user just switched off.
    """
    indexable = set(index_kinds)
    forwardable = set(coworker_kinds)
    now = _now()
    with _get_session(dsn) as db:
        device = db.get(Device, device_id)
        if device is None:
            raise DeviceError("Unknown device", status_code=404)
        enabled = set(device.enabled_categories or [])
        device.last_seen = now

        valid: list[dict[str, Any]] = []
        rejected = 0
        seen: set[str] = set()
        for raw in events:
            try:
                kind = str(raw.get("kind") or "").strip().lower()
                key = str(raw.get("dedup_key") or "").strip()[:_MAX_KEY]
                if kind not in EVENT_KINDS or kind not in enabled or not key or key in seen:
                    rejected += 1
                    continue
                payload = raw.get("payload") or {}
                if not isinstance(payload, dict):
                    raise ValueError("payload must be an object")
                text = raw.get("text")
                valid.append(
                    {
                        "kind": kind,
                        "key": key,
                        "ts": parse_ts(raw.get("ts")),
                        "payload": payload,
                        "text": event_text(kind, payload, text if isinstance(text, str) else None),
                    }
                )
                seen.add(key)
            except (ValueError, TypeError, AttributeError, OverflowError):
                rejected += 1

        existing: set[str] = set()
        keys = [item["key"] for item in valid]
        for start in range(0, len(keys), 500):
            existing.update(
                db.scalars(
                    select(PhoneEvent.dedup_key).where(
                        PhoneEvent.device_id == device_id,
                        PhoneEvent.dedup_key.in_(keys[start : start + 500]),
                    )
                ).all()
            )

        to_index: list[tuple[str, str, str]] = []
        to_coworker: list[tuple[str, str, str]] = []
        accepted = 0
        for item in valid:
            if item["key"] in existing:
                continue
            event_id = uuid.uuid4().hex
            db.add(
                PhoneEvent(
                    id=event_id,
                    device_id=device_id,
                    kind=item["kind"],
                    ts=item["ts"],
                    dedup_key=item["key"],
                    payload=item["payload"],
                    text=item["text"] or None,
                    received_at=now,
                )
            )
            accepted += 1
            ref = source_ref(device_id, event_id)
            if item["kind"] in indexable and item["text"]:
                to_index.append((f"phone_{item['kind']}", ref, item["text"]))
            if item["text"] and coworker_eligible(item["kind"], item["payload"], forwardable):
                to_coworker.append((f"phone_{item['kind']}", ref, item["text"]))
        duplicates = len(valid) - accepted

    if retention_days > 0:
        pruned = purge_events(dsn, device_id, older_than=now - timedelta(days=retention_days))
        if pruned:
            logger.info("phone bridge: pruned %d events for device %s", pruned, device_id)
    return IngestResult(
        accepted=accepted,
        duplicates=duplicates,
        rejected=rejected,
        to_index=to_index,
        to_coworker=to_coworker,
    )


def index_events(cfg: Any, items: list[tuple[str, str, str]]) -> int:
    """Embed ingested event text into knowledge_chunks (best effort)."""
    from pawn_core.knowledge_index import index_text  # noqa: PLC0415

    stored = 0
    for kind, ref, text in items:
        try:
            stored += index_text(cfg, source_kind=kind, source_ref=ref, text=text)
        except Exception as exc:  # noqa: BLE001 — embedding backend may be down
            logger.warning("phone bridge: indexing %s failed: %s", ref, exc)
    return stored


async def forward_to_coworker(cfg: Any, items: list[tuple[str, str, str]]) -> int:
    """Run the coworker pipeline on new missed calls / received SMS (best effort).

    Requires ``coworker.enabled``; each event is its own source so triage and
    re-runs stay per event. Returns how many events were processed.
    """
    coworker = getattr(cfg, "coworker", None)
    if not items or not getattr(coworker, "enabled", False):
        return 0
    from pawn_agent.core.coworker.pipeline import process_source  # noqa: PLC0415

    done = 0
    for kind, ref, text in items:
        try:
            await process_source(cfg, source_kind=kind, source_ref=ref, source_text=text)
            done += 1
        except Exception as exc:  # noqa: BLE001 — model / vault may be down
            logger.warning("phone bridge: coworker pass for %s failed: %s", ref, exc)
    return done
