"""Voice threads: which assistant holds a ``voice:`` conversation, hand-off,
phone-approved full access, and the live mirror into portal chats.

Owns: ``voice_threads`` and ``access_requests`` rows.
Does not own: the voice turn itself (``api_server.pawn_voice``) or the command
queue (``device_commands``), which delivers ``access_approve`` to the phone.

All functions are synchronous; the API layer drives them with
``asyncio.to_thread``. See docs/ASSISTANTS.md and docs/VOICE_PROTOCOL.md.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from sqlalchemy import func, select

from pawn_agent.utils.db import AccessRequest, Device, VoiceThread, _get_session

logger = logging.getLogger(__name__)

VOICE_FOLDER = "Voice"
_SNIPPET = 600


class VoiceThreadError(Exception):
    def __init__(self, message: str, *, status_code: int = 422) -> None:
        super().__init__(message)
        self.status_code = status_code


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() if value else None


def _settings(cfg: Any) -> Any:
    return getattr(cfg, "devices", None)


def thread_view(row: VoiceThread, device_name: Optional[str] = None) -> dict[str, Any]:
    return {
        "conversation": row.conversation,
        "device_id": row.device_id,
        "device_name": device_name,
        "access": row.access,
        "lang": row.lang,
        "state": row.state,
        "turns": row.turns,
        "last_user": row.last_user,
        "last_answer": row.last_answer,
        "portal_chat_id": row.portal_chat_id,
        "started_at": _iso(row.started_at),
        "last_turn_at": _iso(row.last_turn_at),
    }


# ── Turns ─────────────────────────────────────────────────────────────────────


def record_turn(
    cfg: Any,
    conversation: str,
    *,
    device_id: Optional[str],
    access: str,
    lang: str,
    user: str,
    answer: str,
    display: Optional[str] = None,
    run_id: Optional[str] = None,
    ended: bool = False,
) -> None:
    """Upsert the thread after a turn and mirror it to the portal (best effort)."""
    now = _now()
    dsn = cfg.db_dsn
    device_name: Optional[str] = None
    try:
        with _get_session(dsn) as db:
            device = db.get(Device, device_id) if device_id else None
            if device_id and device is None:
                device_id = None  # unknown id: keep the turn, drop the link
            device_name = device.name if device is not None else None
            row = db.get(VoiceThread, conversation)
            if row is None:
                row = VoiceThread(conversation=conversation, started_at=now, turns=0)
                db.add(row)
            if device_id:
                row.device_id = device_id
            row.access = access
            row.lang = lang
            row.state = "ended" if ended else "active"
            row.turns = int(row.turns or 0) + 1
            row.last_user = (user or "")[:_SNIPPET]
            row.last_answer = (answer or "")[:_SNIPPET]
            row.last_turn_at = now
            chat_id = row.portal_chat_id
    except Exception as exc:  # noqa: BLE001 — older schema / DB hiccup must not break a turn
        logger.warning("voice thread %s: not recorded: %s", conversation, exc)
        return
    if not bool(getattr(_settings(cfg), "mirror_voice_to_portal", True)):
        return
    try:
        new_chat = mirror_to_portal(
            dsn,
            conversation,
            chat_id=chat_id,
            device_name=device_name,
            user=user,
            answer=answer,
            display=display,
            run_id=run_id,
            at=now,
        )
        if new_chat and new_chat != chat_id:
            with _get_session(dsn) as db:
                row = db.get(VoiceThread, conversation)
                if row is not None:
                    row.portal_chat_id = new_chat
    except Exception as exc:  # noqa: BLE001
        logger.warning("voice thread %s: portal mirror failed: %s", conversation, exc)


def end_thread(dsn: str, conversation: str) -> None:
    try:
        with _get_session(dsn) as db:
            row = db.get(VoiceThread, conversation)
            if row is not None:
                row.state = "ended"
    except Exception as exc:  # noqa: BLE001
        logger.warning("voice thread %s: end failed: %s", conversation, exc)


def mirror_to_portal(
    dsn: str,
    conversation: str,
    *,
    chat_id: Optional[str],
    device_name: Optional[str],
    user: str,
    answer: str,
    display: Optional[str] = None,
    run_id: Optional[str] = None,
    at: Optional[datetime] = None,
) -> str:
    """Append the user + assistant messages to the portal chat for *conversation*."""
    from pawn_server.core.portal_api import _title_from  # noqa: PLC0415
    from pawn_server.core.portal_import import _ensure_folder  # noqa: PLC0415
    from pawn_server.core.portal_models import PortalChat, PortalMessage  # noqa: PLC0415

    now = at or _now()
    with _get_session(dsn) as db:
        chat = db.get(PortalChat, chat_id) if chat_id else None
        if chat is None:
            chat = db.scalar(select(PortalChat).where(PortalChat.conversation_key == conversation))
        if chat is None:
            tags = ["voice"] + ([device_name] if device_name else [])
            chat = PortalChat(
                id=uuid.uuid4().hex,
                conversation_key=conversation,
                title=_title_from(user) or "Voice conversation",
                folder_id=_ensure_folder(db, VOICE_FOLDER),
                tags=tags,
                title_auto=True,
                tags_auto=False,
                created_at=now,
                updated_at=now,
            )
            db.add(chat)
            db.flush()
        elif device_name and device_name not in (chat.tags or []):
            chat.tags = [*(chat.tags or []), device_name]
        seq = int(db.scalar(select(func.max(PortalMessage.seq)).where(PortalMessage.chat_id == chat.id)) or 0)
        origin = {"source": "voice", "device": device_name}
        db.add(
            PortalMessage(
                id=uuid.uuid4().hex,
                chat_id=chat.id,
                seq=seq + 1,
                role="user",
                content=user,
                context={"origin": origin},
                created_at=now,
            )
        )
        body = answer + (f"\n\n{display}" if display else "")
        db.add(
            PortalMessage(
                id=uuid.uuid4().hex,
                chat_id=chat.id,
                seq=seq + 2,
                role="assistant",
                content=body,
                context={"origin": origin},
                run_id=run_id,
                created_at=now,
            )
        )
        chat.last_message_at = now
        chat.updated_at = now
        return str(chat.id)


# ── Hand-off ──────────────────────────────────────────────────────────────────


def active_thread(
    dsn: str, *, exclude_device: Optional[str] = None, window_s: float = 600.0
) -> Optional[dict[str, Any]]:
    """Newest ``active`` thread with a turn in the last *window_s* seconds."""
    since = _now() - timedelta(seconds=window_s)
    with _get_session(dsn) as db:
        stmt = (
            select(VoiceThread)
            .where(VoiceThread.state == "active", VoiceThread.last_turn_at >= since)
            .order_by(VoiceThread.last_turn_at.desc())
        )
        for row in db.scalars(stmt):
            if exclude_device and row.device_id == exclude_device:
                continue
            device = db.get(Device, row.device_id) if row.device_id else None
            return thread_view(row, device.name if device else None)
    return None


def claim_thread(
    dsn: str, conversation: str, device_id: str, *, window_s: float = 600.0
) -> dict[str, Any]:
    """Move *conversation* to *device_id*. Access always drops back to ``chat``."""
    now = _now()
    with _get_session(dsn) as db:
        device = db.get(Device, device_id)
        if device is None:
            raise VoiceThreadError("Unknown device", status_code=404)
        row = db.get(VoiceThread, conversation)
        if row is None:
            raise VoiceThreadError("Unknown conversation", status_code=404)
        if row.state == "ended":
            raise VoiceThreadError("That conversation has ended", status_code=409)
        if row.last_turn_at and (now - row.last_turn_at).total_seconds() > window_s:
            raise VoiceThreadError("That conversation is too old to resume", status_code=409)
        previous = db.get(Device, row.device_id) if row.device_id else None
        row.device_id = device_id
        row.access = "chat"
        row.state = "active"
        row.last_turn_at = now
        view = thread_view(row, device.name)
        view["from_device"] = previous.name if previous is not None else None
        return view


# ── Phone-approved full access ────────────────────────────────────────────────


def _approver(db: Any, requester_id: str) -> Optional[Device]:
    rows = db.scalars(select(Device).order_by(Device.created_at)).all()
    phones = [
        row
        for row in rows
        if row.id != requester_id
        and (row.kind or "phone") == "phone"
        and "access_approve" in (row.capabilities or [])
    ]
    if not phones:
        return None
    return max(phones, key=lambda row: row.presence_at or row.last_seen or datetime.min)


def request_view(row: AccessRequest, approver_name: Optional[str] = None) -> dict[str, Any]:
    return {
        "id": row.id,
        "requester_device_id": row.requester_device_id,
        "approver_device_id": row.approver_device_id,
        "approver_name": approver_name,
        "conversation": row.conversation,
        "minutes": row.minutes,
        "status": row.status,
        "message": row.message,
        "created_at": _iso(row.created_at),
        "expires_at": _iso(row.expires_at),
        "access_until": _iso(row.access_until),
    }


def create_access_request(
    cfg: Any,
    requester_device_id: str,
    *,
    conversation: Optional[str] = None,
    minutes: int = 15,
    push: Any = None,
) -> dict[str, Any]:
    """Queue ``access_approve`` on the user's phone; returns immediately (pending)."""
    from pawn_server.core import device_commands  # noqa: PLC0415

    settings = _settings(cfg)
    ttl = float(getattr(settings, "access_request_timeout_s", 60.0))
    cap = int(getattr(settings, "access_max_minutes", 30))
    minutes = max(1, min(int(minutes or 15), cap))
    request_id = uuid.uuid4().hex
    with _get_session(cfg.db_dsn) as db:
        requester = db.get(Device, requester_device_id)
        if requester is None:
            raise VoiceThreadError("Unknown device", status_code=404)
        approver = _approver(db, requester_device_id)
        if approver is None:
            raise VoiceThreadError(
                "No phone can approve access (turn on 'Approve other assistants' in the app).",
                status_code=409,
            )
        requester_name, approver_id = requester.name, approver.id
    try:
        _, command_id, _error = device_commands.queue_command(
            cfg.db_dsn,
            "access_approve",
            {"requester": requester_name, "minutes": minutes, "request_id": request_id},
            device=approver_id,
            ttl_s=ttl,
            push=push,
        )
    except device_commands.CommandError as exc:
        raise VoiceThreadError(str(exc), status_code=exc.status_code) from exc
    now = _now()
    with _get_session(cfg.db_dsn) as db:
        row = AccessRequest(
            id=request_id,
            requester_device_id=requester_device_id,
            approver_device_id=approver_id,
            conversation=conversation,
            minutes=minutes,
            status="pending",
            command_id=command_id,
            created_at=now,
            expires_at=now + timedelta(seconds=ttl),
        )
        db.add(row)
        db.flush()
        approver_row = db.get(Device, approver_id)
        return request_view(row, approver_row.name if approver_row else None)


_COMMAND_TO_REQUEST = {"ok": "approved", "denied": "denied", "error": "error", "expired": "expired"}


def get_access_request(dsn: str, request_id: str) -> dict[str, Any]:
    """Current status; folds in the phone's answer to the ``access_approve`` command."""
    from pawn_server.core import device_commands  # noqa: PLC0415

    with _get_session(dsn) as db:
        row = db.get(AccessRequest, request_id)
        if row is None:
            raise VoiceThreadError("Unknown access request", status_code=404)
        pending_command = row.command_id if row.status == "pending" else None
    # Read the command in its own session (it may expire the row itself).
    command = device_commands.get_command(dsn, pending_command) if pending_command else None
    with _get_session(dsn) as db:
        row = db.get(AccessRequest, request_id)
        assert row is not None
        if row.status == "pending" and pending_command:
            status = (command or {}).get("status", "expired")
            if status in _COMMAND_TO_REQUEST:
                row.status = _COMMAND_TO_REQUEST[status]
                result = (command or {}).get("result") or {}
                row.message = str(result.get("message") or "")[:500] or None
                if row.status == "approved":
                    row.access_until = _now() + timedelta(minutes=row.minutes)
            elif row.expires_at and row.expires_at < _now():
                row.status = "expired"
        approver = db.get(Device, row.approver_device_id) if row.approver_device_id else None
        return request_view(row, approver.name if approver else None)


def list_access_requests(
    dsn: str, *, device_id: Optional[str] = None, pending_only: bool = False, limit: int = 20
) -> list[dict[str, Any]]:
    with _get_session(dsn) as db:
        stmt = select(AccessRequest).order_by(AccessRequest.created_at.desc()).limit(limit)
        if device_id:
            stmt = stmt.where(
                (AccessRequest.requester_device_id == device_id)
                | (AccessRequest.approver_device_id == device_id)
            )
        if pending_only:
            stmt = stmt.where(AccessRequest.status == "pending")
        ids = [row.id for row in db.scalars(stmt)]
    return [get_access_request(dsn, rid) for rid in ids]


def list_threads(dsn: str, *, device_id: Optional[str] = None, limit: int = 20) -> list[dict[str, Any]]:
    with _get_session(dsn) as db:
        stmt = select(VoiceThread).order_by(VoiceThread.last_turn_at.desc()).limit(limit)
        if device_id:
            stmt = stmt.where(VoiceThread.device_id == device_id)
        names = {row.id: row.name for row in db.scalars(select(Device))}
        return [thread_view(row, names.get(row.device_id or "")) for row in db.scalars(stmt)]
