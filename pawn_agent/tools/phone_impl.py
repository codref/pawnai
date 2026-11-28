"""Read tools over phone-bridge data stored on the server.

These answer from ``phone_events`` (what the phone already pushed), so they
work while the phone is offline. Live reads and actions go through the device
command queue instead. See docs/PHONE_BRIDGE.md.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from sqlalchemy import select
from sqlalchemy.orm import Session

from pawn_agent.utils.db import Device, PhoneEvent
from pawn_core.database import get_engine

_KINDS = ("call", "sms", "notification", "location", "status")
_SINCE_RE = re.compile(r"^\s*(\d+)\s*([mhdw])\s*$", re.IGNORECASE)
_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def parse_since(raw: str, *, now: Optional[datetime] = None) -> Optional[datetime]:
    """``24h`` / ``7d`` / ``30m`` / ``2w`` or an ISO date → naive UTC cutoff."""
    text = (raw or "").strip()
    if not text:
        return None
    base = now or datetime.now(timezone.utc).replace(tzinfo=None)
    match = _SINCE_RE.match(text)
    if match:
        amount, unit = int(match.group(1)), match.group(2).lower()
        return base - timedelta(**{_UNITS[unit]: amount})
    parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    if parsed.tzinfo is not None:
        parsed = parsed.astimezone(timezone.utc).replace(tzinfo=None)
    return parsed


def _tz(cfg: Any) -> ZoneInfo:
    name = getattr(getattr(cfg, "coworker", None), "timezone", "") or "UTC"
    try:
        return ZoneInfo(name)
    except Exception:  # noqa: BLE001
        return ZoneInfo("UTC")


def _local(ts: datetime, tz: ZoneInfo) -> str:
    return ts.replace(tzinfo=timezone.utc).astimezone(tz).strftime("%Y-%m-%d %H:%M")


def format_event(row: PhoneEvent, tz: ZoneInfo) -> str:
    """One readable line per event (what the model sees)."""
    p = row.payload or {}
    when = _local(row.ts, tz)
    name = p.get("name") or ""
    number = p.get("number") or p.get("address") or ""
    who = f"{name} ({number})" if name and number else (name or number or "unknown")
    if row.kind == "call":
        duration = p.get("duration_s")
        tail = f", {duration}s" if duration else ""
        return f"{when} call {p.get('direction') or ''} {who}{tail}".replace("  ", " ")
    if row.kind == "sms":
        direction = p.get("direction") or "received"
        arrow = "from" if direction == "received" else "to"
        return f"{when} sms {arrow} {who}: {p.get('body') or row.text or ''}"
    if row.kind == "notification":
        app = p.get("app") or p.get("package") or "app"
        parts = " — ".join(str(x) for x in (p.get("title"), p.get("text")) if x)
        return f"{when} notification [{app}] {parts or row.text or ''}"
    if row.kind == "location":
        acc = p.get("accuracy_m")
        tail = f" ±{int(acc)}m" if isinstance(acc, (int, float)) else ""
        place = f" {p['place']}" if p.get("place") else ""
        return f"{when} location {p.get('lat')},{p.get('lon')}{tail}{place}"
    if row.kind == "status":
        bits = []
        if p.get("battery_pct") is not None:
            bits.append(f"battery {p['battery_pct']}%" + (" charging" if p.get("charging") else ""))
        for key in ("network", "ringer"):
            if p.get(key):
                bits.append(f"{key} {p[key]}")
        return f"{when} status {', '.join(bits) or row.text or ''}"
    return f"{when} {row.kind} {row.text or ''}"


def phone_events_impl(
    cfg: Any,
    *,
    kind: str = "",
    since: str = "24h",
    contact: str = "",
    query: str = "",
    limit: int = 30,
    device: str = "",
) -> str:
    """List stored phone events, newest first."""
    kind = (kind or "").strip().lower()
    if kind and kind not in _KINDS:
        return f"Error: --kind must be one of {', '.join(_KINDS)}"
    limit = max(1, min(int(limit or 30), 200))
    cutoff = parse_since(since)
    tz = _tz(cfg)
    with Session(get_engine(cfg.db_dsn)) as db:
        devices = {d.id: d for d in db.scalars(select(Device)).all()}
        if not devices:
            return "No phone is registered with pawn-server (phone bridge is off)."
        stmt = select(PhoneEvent).order_by(PhoneEvent.ts.desc())
        if device:
            matches = [d.id for d in devices.values() if device in (d.id, d.name)]
            if not matches:
                return f"Error: unknown device {device!r}"
            stmt = stmt.where(PhoneEvent.device_id.in_(matches))
        if kind:
            stmt = stmt.where(PhoneEvent.kind == kind)
        if cutoff is not None:
            stmt = stmt.where(PhoneEvent.ts >= cutoff)
        if query.strip():
            stmt = stmt.where(PhoneEvent.text.ilike(f"%{query.strip()}%"))
        # Contact matching is done in Python (payload is JSON, portable across PG/SQLite).
        fetch = limit if not contact.strip() else limit * 10
        rows = list(db.scalars(stmt.limit(fetch)).all())
        if contact.strip():
            needle = contact.strip().lower()
            digits = re.sub(r"\D", "", needle)
            kept = []
            for row in rows:
                p = row.payload or {}
                name = str(p.get("name") or "").lower()
                number = re.sub(r"\D", "", str(p.get("number") or p.get("address") or ""))
                if needle in name or (digits and len(digits) >= 4 and digits in number):
                    kept.append(row)
            rows = kept[:limit]

        enabled = sorted({k for d in devices.values() for k in (d.enabled_categories or [])})
        scope = kind or "phone"
        window = f" since {since}" if since else ""
        if not rows:
            hint = ""
            if kind and kind not in enabled:
                hint = f" (sharing of '{kind}' is turned off on the phone)"
            return f"No {scope} events{window}{hint}."
        lines = [format_event(row, tz) for row in rows]
        if len(devices) > 1 and not device:
            lines = [f"[{devices[row.device_id].name}] {line}" for row, line in zip(rows, lines)]
        header = f"{len(rows)} {scope} event(s){window}, newest first ({tz.key}):"
        return "\n".join([header, *lines])


def phone_devices_impl(cfg: Any) -> str:
    """Registered phones, what they share, and when they were last seen."""
    tz = _tz(cfg)
    with Session(get_engine(cfg.db_dsn)) as db:
        rows = db.scalars(select(Device).order_by(Device.created_at)).all()
        if not rows:
            return "No phone is registered with pawn-server (phone bridge is off)."
        out = []
        for row in rows:
            shared = ", ".join(row.enabled_categories or []) or "nothing"
            seen = _local(row.last_seen, tz) if row.last_seen else "never"
            kind = row.kind or "phone"
            live = "yes" if (row.unifiedpush_endpoint or kind == "desktop") else "no"
            tools = ", ".join(row.capabilities or []) or "none"
            presence = row.presence or {}
            here = ""
            if row.presence_at is not None:
                flags = [k for k in ("headset", "on_call", "in_session") if presence.get(k)]
                here = f"; presence {presence.get('state', '?')}" + (
                    f" ({', '.join(flags)})" if flags else ""
                ) + f" at {_local(row.presence_at, tz)}"
            out.append(
                f"- {row.name} ({kind}, id {row.id}): shares {shared}; runs {tools}; "
                f"last seen {seen}{here}; live tools {live}"
            )
        return "\n".join(out)


__all__ = ["format_event", "parse_since", "phone_devices_impl", "phone_events_impl"]
