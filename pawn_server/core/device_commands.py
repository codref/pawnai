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
        summary="show a notification on the phone",
    ),
}


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
    return clean


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
    }


def resolve_device(dsn: str, device: str = "") -> Device:
    """Pick a device by id/name; with one device registered, *device* may be empty."""
    with _get_session(dsn) as db:
        rows = list(db.scalars(select(Device).order_by(Device.created_at)).all())
        for row in rows:
            db.expunge(row)
    if not rows:
        raise CommandError("No phone is registered with pawn-server (phone bridge is off).", status_code=404)
    if device:
        matches = [row for row in rows if device in (row.id, row.name)]
        if not matches:
            raise CommandError(f"Unknown device {device!r}", status_code=404)
        return matches[0]
    if len(rows) > 1:
        names = ", ".join(row.name for row in rows)
        raise CommandError(f"Several phones are registered ({names}); pass --device NAME")
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
            f"The phone '{device.name}' does not offer {tool} "
            "(turn it on in pawn-assistant → Phone bridge)."
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


def run_command(
    dsn: str,
    tool: str,
    args: Optional[dict[str, Any]] = None,
    *,
    device: str = "",
    timeout_s: float = 45.0,
    poll_s: float = 0.5,
    push: Optional[Callable[[str], Optional[str]]] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Queue *tool* on the phone, wake it, wait for the answer.

    Returns the command view; ``status`` is one of ``ok`` / ``denied`` /
    ``error`` / ``expired`` / ``delivered`` (phone took it but did not answer
    in time — an action may still complete on the phone).
    """
    clean = validate_args(tool, args)
    target = resolve_device(dsn, device)
    if not target.unifiedpush_endpoint:
        raise CommandError(
            f"The phone '{target.name}' has no push channel yet "
            "(install a UnifiedPush distributor such as ntfy on the phone)."
        )
    command_id = create_command(dsn, target, tool, clean, ttl_s=timeout_s)
    error = (push or send_push)(target.unifiedpush_endpoint)
    if error:
        logger.warning("phone bridge: %s for device %s", error, target.id)
    deadline = time.monotonic() + timeout_s
    while True:
        with _get_session(dsn) as db:
            row = db.get(DeviceCommand, command_id)
            assert row is not None
            if row.status in FINAL:
                return command_view(row)
            if time.monotonic() >= deadline:
                if row.status == "pending":
                    row.status = "expired"
                    row.completed_at = _now()
                    if error:
                        row.result = {"message": error}
                return command_view(row)
        sleep(poll_s)


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
        if device is not None:
            device.last_seen = _now()
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
            return "Notification shown on the phone."
        return f"{tool}: " + json.dumps(result, ensure_ascii=False)
    if status == "denied":
        return f"The user declined {tool} on the phone." + (f" ({note})" if note else "")
    if status == "error":
        return f"Error: the phone could not run {tool}: {note or 'unknown error'}"
    if status == "delivered":
        tail = " It may still complete on the phone." if TOOLS[tool].action else ""
        return f"The phone received {tool} but did not answer in time.{tail}"
    return "Error: the phone did not answer (it may be offline or its push channel is down)." + (
        f" ({note})" if note else ""
    )
