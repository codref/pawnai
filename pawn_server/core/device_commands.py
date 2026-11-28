"""Phone bridge command queue: live tools that run on the phone.

Flow (see docs/PHONE_BRIDGE.md):

1. A ``phone_*`` CliTool (subprocess) calls :func:`run_command`, which inserts
   a ``device_commands`` row and sends a content-free UnifiedPush wake-up to
   the device's endpoint (an ntfy topic URL).
2. The phone fetches ``GET /v1/devices/{id}/commands/pending`` (rows become
   ``delivered``), runs each command — actions only after an on-phone
   biometric confirmation — and posts ``…/commands/{cid}/result``.
3. :func:`run_command` polls the row until it is answered or the budget
   (``devices.command_timeout_s``) runs out.

The push body carries no data: ntfy topics may be on a shared server, so the
phone always pulls the command over the authenticated API.

State is in the database (not in-process) because CliTools run as separate
processes from the API server that receives the phone's result.
"""

from __future__ import annotations

import json
import logging
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Optional

from sqlalchemy import select

from pawn_agent.utils.db import Device, DeviceCommand, _get_session

logger = logging.getLogger(__name__)

PUSH_BODY = b'{"type":"cmd"}'
FINAL = ("ok", "denied", "error", "expired")
_RESULT_MAX = 20_000


@dataclass(frozen=True)
class ToolSpec:
    name: str
    action: bool  # True → the phone must get an explicit biometric confirmation
    required: tuple[str, ...] = ()
    optional: tuple[str, ...] = ()
    max_len: int = 1000
    summary: str = ""


TOOLS: dict[str, ToolSpec] = {
    "location_now": ToolSpec("location_now", action=False, summary="current location fix"),
    "status_now": ToolSpec("status_now", action=False, summary="battery / network / ringer now"),
    "sms_send": ToolSpec(
        "sms_send",
        action=True,
        required=("to", "body"),
        summary="send an SMS (confirmed on the phone)",
    ),
    "notify": ToolSpec(
        "notify",
        action=False,
        required=("text",),
        optional=("title",),
        max_len=500,
        summary="show a notification on the device",
    ),
    # Assistants (docs/ASSISTANTS.md): any voice assistant, phone or desktop.
    "speak": ToolSpec(
        "speak",
        action=False,
        required=("text",),
        max_len=500,
        summary="say a sentence aloud on the assistant",
    ),
    "timer_set": ToolSpec(
        "timer_set",
        action=False,
        required=("seconds",),
        optional=("label",),
        max_len=120,
        summary="start a countdown timer on the assistant",
    ),
    "access_approve": ToolSpec(
        "access_approve",
        action=True,
        required=("requester", "minutes"),
        optional=("request_id",),
        max_len=120,
        summary="approve full voice access for another assistant (confirmed on the phone)",
    ),
}

#: Device kinds. Phones are woken over UnifiedPush; desktops hold an SSE stream.
DEVICE_KINDS: tuple[str, ...] = ("phone", "desktop")

_MAX_TIMER_S = 24 * 3600


class CommandError(Exception):
    def __init__(self, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def validate_args(tool: str, args: Optional[dict[str, Any]]) -> dict[str, str]:
    spec = TOOLS.get(tool)
    if spec is None:
        raise CommandError(f"Unknown phone tool {tool!r}")
    raw = args or {}
    if not isinstance(raw, dict):
        raise CommandError("args must be an object")
    allowed = set(spec.required) | set(spec.optional)
    unknown = sorted(set(raw) - allowed)
    if unknown:
        raise CommandError(f"Unexpected args for {tool}: {', '.join(unknown)}")
    clean: dict[str, str] = {}
    for key in allowed:
        value = raw.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if len(text) > spec.max_len:
            raise CommandError(f"{key} is longer than {spec.max_len} characters")
        if text:
            clean[key] = text
    missing = [key for key in spec.required if key not in clean]
    if missing:
        raise CommandError(f"Missing args for {tool}: {', '.join(missing)}")
    if tool == "timer_set":
        clean["seconds"] = str(_positive_int(clean["seconds"], "seconds", _MAX_TIMER_S))
    if tool == "access_approve":
        clean["minutes"] = str(_positive_int(clean["minutes"], "minutes", 24 * 60))
    return clean


def _positive_int(raw: str, key: str, limit: int) -> int:
    try:
        value = int(float(raw))
    except ValueError as exc:
        raise CommandError(f"{key} must be a number") from exc
    if value <= 0 or value > limit:
        raise CommandError(f"{key} must be between 1 and {limit}")
    return value


def command_view(row: DeviceCommand) -> dict[str, Any]:
    spec = TOOLS.get(row.tool)
    return {
        "id": row.id,
        "tool": row.tool,
        "args": dict(row.args or {}),
        "action": bool(spec and spec.action),
        "status": row.status,
        "result": row.result,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        "expires_at": row.expires_at.isoformat() if row.expires_at else None,
        "completed_at": row.completed_at.isoformat() if row.completed_at else None,
    }


def _is_fresh(row: Device, fresh_s: float, now: datetime) -> bool:
    if row.presence_at is None:
        return False
    state = str((row.presence or {}).get("state") or "active")
    return state != "away" and (now - row.presence_at).total_seconds() <= fresh_s


def pick_auto(rows: list[Device], *, tool: str = "", fresh_s: float = 600.0) -> Optional[Device]:
    """The device the user is most likely near that can run *tool*.

    Order: a fresh ``active`` presence, then a fresh ``idle`` one (newest
    report first), then the most recently seen device.
    """
    able = [row for row in rows if not tool or tool in (row.capabilities or [])]
    if not able:
        return None
    now = _now()
    fresh = [row for row in able if _is_fresh(row, fresh_s, now)]
    if fresh:
        def rank(row: Device) -> tuple[int, datetime]:
            state = str((row.presence or {}).get("state") or "active")
            return (1 if state == "active" else 0, row.presence_at or datetime.min)

        return max(fresh, key=rank)
    return max(able, key=lambda row: row.last_seen or row.created_at or datetime.min)


def resolve_device(
    dsn: str,
    device: str = "",
    *,
    tool: str = "",
    fresh_s: float = 600.0,
    kind: str = "",
) -> Device:
    """Pick a device by id/name; with one device registered, *device* may be empty.

    ``device="auto"`` picks by presence among the devices offering *tool*
    (see :func:`pick_auto`). *kind* narrows the candidates when *device* is
    empty or ``auto`` (``phone_*`` tools pass ``kind="phone"``).
    """
    with _get_session(dsn) as db:
        rows = list(db.scalars(select(Device).order_by(Device.created_at)).all())
        for row in rows:
            db.expunge(row)
    if kind and device in ("", "auto"):
        rows = [row for row in rows if (row.kind or "phone") == kind]
    if not rows:
        raise CommandError("No assistant is registered with pawn-server (phone bridge is off).", status_code=404)
    if device == "auto":
        picked = pick_auto(rows, tool=tool, fresh_s=fresh_s)
        if picked is None:
            raise CommandError(f"No registered assistant offers {tool}.", status_code=404)
        return picked
    if device:
        matches = [row for row in rows if device in (row.id, row.name)]
        if not matches:
            raise CommandError(f"Unknown device {device!r}", status_code=404)
        return matches[0]
    if len(rows) > 1:
        names = ", ".join(row.name for row in rows)
        raise CommandError(f"Several devices are registered ({names}); pass --device NAME or auto")
    return rows[0]


def send_push(endpoint: str, *, timeout: float = 10.0) -> Optional[str]:
    """POST the wake-up to a UnifiedPush endpoint. Returns an error string or None."""
    request = urllib.request.Request(
        endpoint,
        data=PUSH_BODY,
        headers={"Content-Type": "application/json", "TTL": "60", "Urgency": "high"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status >= 300:
                return f"push endpoint returned {response.status}"
    except (urllib.error.URLError, OSError) as exc:
        return f"push failed: {exc}"
    return None


def create_command(
    dsn: str, device: Device, tool: str, args: dict[str, str], *, ttl_s: float
) -> str:
    if tool not in (device.capabilities or []):
        raise CommandError(
            f"The {device.kind or 'phone'} '{device.name}' does not offer {tool} "
            "(turn it on in the assistant's settings)."
        )
    command_id = uuid.uuid4().hex
    now = _now()
    with _get_session(dsn) as db:
        db.add(
            DeviceCommand(
                id=command_id,
                device_id=device.id,
                tool=tool,
                args=args,
                status="pending",
                created_at=now,
                expires_at=now + timedelta(seconds=ttl_s),
            )
        )
    return command_id


def queue_command(
    dsn: str,
    tool: str,
    args: Optional[dict[str, Any]] = None,
    *,
    device: str = "",
    ttl_s: float = 45.0,
    fresh_s: float = 600.0,
    kind: str = "",
    push: Optional[Callable[[str], Optional[str]]] = None,
) -> tuple[Device, str, Optional[str]]:
    """Insert a command and wake the device. Returns ``(device, command_id, push_error)``.

    Desktops (``kind="desktop"``) hold ``GET …/commands/stream`` open, so they
    need no push; phones need a UnifiedPush endpoint.
    """
    clean = validate_args(tool, args)
    target = resolve_device(dsn, device, tool=tool, fresh_s=fresh_s, kind=kind)
    streamed = (target.kind or "phone") == "desktop"
    if not streamed and not target.unifiedpush_endpoint:
        raise CommandError(
            f"The phone '{target.name}' has no push channel yet "
            "(install a UnifiedPush distributor such as ntfy on the phone)."
        )
    command_id = create_command(dsn, target, tool, clean, ttl_s=ttl_s)
    error: Optional[str] = None
    if target.unifiedpush_endpoint:
        error = (push or send_push)(target.unifiedpush_endpoint)
        if error and not streamed:
            logger.warning("phone bridge: %s for device %s", error, target.id)
        elif error:
            error = None  # the stream delivers it anyway
    return target, command_id, error


def run_command(
    dsn: str,
    tool: str,
    args: Optional[dict[str, Any]] = None,
    *,
    device: str = "",
    timeout_s: float = 45.0,
    poll_s: float = 0.5,
    fresh_s: float = 600.0,
    kind: str = "",
    push: Optional[Callable[[str], Optional[str]]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Queue *tool* on the device, wake it, wait for the answer.

    Returns the command view; ``status`` is one of ``ok`` / ``denied`` /
    ``error`` / ``expired`` / ``delivered`` (device took it but did not answer
    in time — an action may still complete on the phone).
    """
    target, command_id, error = queue_command(
        dsn, tool, args, device=device, ttl_s=timeout_s, fresh_s=fresh_s, kind=kind, push=push
    )
    deadline = time.monotonic() + timeout_s
    while True:
        with _get_session(dsn) as db:
            row = db.get(DeviceCommand, command_id)
            assert row is not None
            if row.status in FINAL:
                return {**command_view(row), "device": target.name}
            if time.monotonic() >= deadline:
                if row.status == "pending":
                    row.status = "expired"
                    row.completed_at = _now()
                    if error:
                        row.result = {"message": error}
                return {**command_view(row), "device": target.name}
        sleep(poll_s)


def get_command(dsn: str, command_id: str) -> Optional[dict[str, Any]]:
    with _get_session(dsn) as db:
        row = db.get(DeviceCommand, command_id)
        if row is None:
            return None
        if row.status == "pending" and row.expires_at and row.expires_at < _now():
            row.status = "expired"
            row.completed_at = _now()
        return command_view(row)


def list_commands(dsn: str, device_id: str, *, limit: int = 20) -> list[dict[str, Any]]:
    with _get_session(dsn) as db:
        rows = db.scalars(
            select(DeviceCommand)
            .where(DeviceCommand.device_id == device_id)
            .order_by(DeviceCommand.created_at.desc())
            .limit(max(1, min(limit, 200)))
        ).all()
        return [command_view(row) for row in rows]


# ── Phone side ────────────────────────────────────────────────────────────────


def expire_stale(dsn: str, device_id: str) -> int:
    now = _now()
    with _get_session(dsn) as db:
        rows = db.scalars(
            select(DeviceCommand).where(
                DeviceCommand.device_id == device_id,
                DeviceCommand.status == "pending",
                DeviceCommand.expires_at < now,
            )
        ).all()
        for row in rows:
            row.status = "expired"
            row.completed_at = now
        return len(rows)


def take_pending(dsn: str, device_id: str) -> list[dict[str, Any]]:
    """Hand the phone its unexpired pending commands, marking them delivered."""
    expire_stale(dsn, device_id)
    with _get_session(dsn) as db:
        device = db.get(Device, device_id)
        now = _now()
        # The desktop stream polls every second: only refresh last_seen now and then.
        if device is not None and (
            device.last_seen is None or (now - device.last_seen).total_seconds() >= 30
        ):
            device.last_seen = now
        rows = db.scalars(
            select(DeviceCommand)
            .where(DeviceCommand.device_id == device_id, DeviceCommand.status == "pending")
            .order_by(DeviceCommand.created_at)
        ).all()
        for row in rows:
            row.status = "delivered"
        return [command_view(row) for row in rows]


def post_result(
    dsn: str,
    device_id: str,
    command_id: str,
    *,
    status: str,
    result: Optional[dict[str, Any]] = None,
    message: str = "",
) -> dict[str, Any]:
    if status not in ("ok", "denied", "error"):
        raise CommandError("status must be ok, denied or error")
    payload = dict(result or {})
    if message:
        payload["message"] = str(message)[:1000]
    if len(json.dumps(payload, default=str)) > _RESULT_MAX:
        raise CommandError("result is too large", status_code=413)
    with _get_session(dsn) as db:
        row = db.get(DeviceCommand, command_id)
        if row is None or row.device_id != device_id:
            raise CommandError("Unknown command", status_code=404)
        if row.status in FINAL:
            raise CommandError(f"Command already {row.status}", status_code=409)
        row.status = status
        row.result = payload
        row.completed_at = _now()
        return command_view(row)


# ── Agent-facing text ─────────────────────────────────────────────────────────


def describe(view: dict[str, Any]) -> str:
    """Readable observation for the model."""
    tool, status = view["tool"], view["status"]
    result = view.get("result") or {}
    note = result.get("message") or ""
    if status == "ok":
        if tool == "location_now":
            acc = result.get("accuracy_m")
            acc_txt = f" ±{int(acc)}m" if isinstance(acc, (int, float)) else ""
            place = f" ({result['place']})" if result.get("place") else ""
            when = f" at {result['ts']}" if result.get("ts") else ""
            return f"Phone location: {result.get('lat')},{result.get('lon')}{acc_txt}{place}{when}"
        if tool == "status_now":
            bits = [f"{k} {v}" for k, v in result.items() if k != "message" and v not in (None, "")]
            return "Phone status: " + (", ".join(bits) or "no data")
        if tool == "sms_send":
            return f"SMS sent to {view['args'].get('to')}."
        if tool == "notify":
            return f"Notification shown on {view.get('device') or 'the device'}."
        if tool == "speak":
            return f"Said aloud on {view.get('device') or 'the assistant'}."
        if tool == "timer_set":
            label = view["args"].get("label")
            return f"Timer of {view['args'].get('seconds')} s started on {view.get('device') or 'the assistant'}" + (
                f" ({label})." if label else "."
            )
        if tool == "access_approve":
            return "Access approved on the phone."
        return f"{tool}: " + json.dumps(result, ensure_ascii=False)
    if status == "denied":
        return f"The user declined {tool} on the phone." + (f" ({note})" if note else "")
    if status == "error":
        return f"Error: the phone could not run {tool}: {note or 'unknown error'}"
    if status == "delivered":
        tail = " It may still complete on the phone." if TOOLS[tool].action else ""
        return f"The phone received {tool} but did not answer in time.{tail}"
    return "Error: the device did not answer (it may be offline or its push channel is down)." + (
        f" ({note})" if note else ""
    )
