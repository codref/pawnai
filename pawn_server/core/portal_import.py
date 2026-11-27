"""Import existing sallm conversations into the portal (``pawn-server portal import``).

sallm keeps every conversation in ``{state_dir}/state.db``. The portal has its
own transcript mirror (``portal_messages``) and only shows chats that have a
``portal_chats`` row. This module creates that row and mirror for sallm
sessions that predate the portal (Obsidian plugin chats, Matrix rooms, the
OpenAI-compatible API, …).

Each imported chat keeps the sallm session id as its ``conversation_key``, so
replying from the portal continues the same sallm memory. Only user/assistant
``chat`` messages are mirrored; tool calls, tool results and ingest rows stay
in sallm. Sessions that already have a portal chat are skipped, so the import
can be re-run safely. See docs/PORTAL.md.
"""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from sqlalchemy import func, select

from pawn_core.database import _get_session
from pawn_server.core.portal_api import _title_from
from pawn_server.core.portal_models import PortalChat, PortalChatFolder, PortalMessage

#: Conversation-key prefixes imported when none are given: interactive chats
#: (plugin / API ``chat:{uuid}``) and Matrix rooms. Note tasks (``note:``),
#: voice, coworker and diarization-session keys are automation, not chats.
DEFAULT_PREFIXES = ("chat:", "matrix:")
DEFAULT_FOLDER = "Imported"


@dataclass
class SallmConversation:
    key: str
    created_at: Optional[datetime]
    updated_at: Optional[datetime]
    # (role, content, created_at, context): user prompts split by split_prompt.
    messages: list[tuple[str, str, Optional[datetime], dict[str, Any]]]


@dataclass
class ImportReport:
    imported: list[tuple[str, str, int]] = field(default_factory=list)  # key, title, n
    skipped_existing: list[str] = field(default_factory=list)
    skipped_empty: list[str] = field(default_factory=list)
    folder_id: Optional[str] = None


# The marker build_chat_prompt (chat_context.py) puts between the user's message
# and the vault context it injects. Stored prompts may have whitespace collapsed.
_CONTEXT_MARKER = re.compile(
    r"\s*---\s*Context from the user's Obsidian vault \(reference material;\s*"
    r"the message above is the request\):\s*"
)
_ACTIVE_NOTE = re.compile(r"^Active note: .*?\(vault path `([^`]+)`\)")


def split_prompt(content: str) -> tuple[str, dict[str, Any]]:
    """Split a stored user prompt into the raw message and its injected context."""
    context: dict[str, Any] = {"imported": True}
    parts = _CONTEXT_MARKER.split(content or "", maxsplit=1)
    if len(parts) < 2:
        return content or "", context
    message, rest = parts[0].rstrip(), parts[1].strip()
    if rest:
        context["prompt_context"] = rest
        active = _ACTIVE_NOTE.match(rest)
        if active:
            context["active_note"] = active.group(1)
    return message, context


def _ts(value: Optional[float]) -> Optional[datetime]:
    if value is None:
        return None
    return datetime.fromtimestamp(float(value), tz=timezone.utc).replace(tzinfo=None)


def matches(key: str, prefixes: Iterable[str]) -> bool:
    """True when *key* starts with one of *prefixes* (``"*"`` matches all)."""
    return any(p == "*" or key.startswith(p) for p in prefixes)


def read_sallm_conversations(
    state_db: Path, prefixes: Iterable[str] = DEFAULT_PREFIXES
) -> list[SallmConversation]:
    """Read matching sessions and their user/assistant chat turns, oldest first."""
    from sallm.state.repository import SessionRepository  # noqa: PLC0415

    prefixes = tuple(prefixes)
    repo = SessionRepository(state_db)
    try:
        m = repo.m
        out: list[SallmConversation] = []
        for sess in m.Session.select().order_by(m.Session.created_at):
            if not matches(sess.id, prefixes):
                continue
            rows = (
                m.Message.select()
                .where(
                    (m.Message.session == sess.id)
                    & (m.Message.kind == "chat")
                    & (m.Message.role.in_(("user", "assistant")))
                )
                .order_by(m.Message.seq)
            )
            msgs = []
            for r in rows:
                if r.role == "user":
                    text, ctx = split_prompt(r.content or "")
                else:
                    text, ctx = r.content or "", {"imported": True}
                msgs.append((r.role, text, _ts(r.created_at), ctx))
            out.append(
                SallmConversation(
                    key=sess.id,
                    created_at=_ts(sess.created_at),
                    updated_at=_ts(sess.updated_at),
                    messages=msgs,
                )
            )
        return out
    finally:
        repo.close()


def _ensure_folder(db: Any, name: str) -> str:
    folder = db.scalar(
        select(PortalChatFolder).where(
            PortalChatFolder.name == name, PortalChatFolder.parent_id.is_(None)
        )
    )
    if folder is not None:
        return str(folder.id)
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    position = db.scalar(select(func.count()).where(PortalChatFolder.parent_id.is_(None)))
    folder = PortalChatFolder(
        id=uuid.uuid4().hex, name=name, position=int(position or 0), created_at=now, updated_at=now
    )
    db.add(folder)
    db.flush()
    return folder.id


def import_conversations(
    dsn: str,
    conversations: Iterable[SallmConversation],
    *,
    folder: Optional[str] = DEFAULT_FOLDER,
    tags: Iterable[str] = (),
    dry_run: bool = False,
) -> ImportReport:
    """Create portal chats + mirrors for *conversations* not yet in the portal."""
    report = ImportReport()
    convs = list(conversations)
    tag_list = [t.strip().strip("#") for t in tags if t.strip().strip("#")]
    with _get_session(dsn) as db:
        existing = set(
            db.scalars(
                select(PortalChat.conversation_key).where(
                    PortalChat.conversation_key.in_([c.key for c in convs] or [""])
                )
            )
        )
        todo: list[SallmConversation] = []
        for conv in convs:
            if conv.key in existing:
                report.skipped_existing.append(conv.key)
            elif not conv.messages:
                report.skipped_empty.append(conv.key)
            else:
                todo.append(conv)
        if dry_run or not todo:
            for conv in todo:
                first = next((m[1] for m in conv.messages if m[0] == "user"), "")
                report.imported.append(
                    (conv.key, _title_from(first) or conv.key, len(conv.messages))
                )
            return report

        folder_id = _ensure_folder(db, folder) if folder else None
        report.folder_id = folder_id
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        for conv in todo:
            chat_id = uuid.uuid4().hex
            first = next((m[1] for m in conv.messages if m[0] == "user"), "")
            title = _title_from(first) or conv.key
            last_at = next((m[2] for m in reversed(conv.messages) if m[2]), None)
            db.add(
                PortalChat(
                    id=chat_id,
                    conversation_key=conv.key,
                    title=title,
                    folder_id=folder_id,
                    tags=list(tag_list),
                    # Let the background model rename/tag on the next answer.
                    title_auto=True,
                    tags_auto=not tag_list,
                    last_message_at=last_at or conv.updated_at,
                    created_at=conv.created_at or now,
                    updated_at=conv.updated_at or now,
                )
            )
            db.flush()
            for seq, (role, content, at, ctx) in enumerate(conv.messages, start=1):
                db.add(
                    PortalMessage(
                        id=uuid.uuid4().hex,
                        chat_id=chat_id,
                        seq=seq,
                        role=role,
                        content=content,
                        context=ctx,
                        created_at=at or conv.created_at or now,
                    )
                )
            report.imported.append((conv.key, title, len(conv.messages)))
    return report


def import_sallm_chats(
    cfg: Any,
    *,
    prefixes: Iterable[str] = DEFAULT_PREFIXES,
    folder: Optional[str] = DEFAULT_FOLDER,
    tags: Iterable[str] = (),
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> ImportReport:
    """Read sallm ``state.db`` under the configured state dir and import it."""
    from pawn_agent.core.sallm_factory import resolve_state_dir  # noqa: PLC0415

    state_db = resolve_state_dir(cfg) / "state.db"
    if not state_db.exists():
        raise FileNotFoundError(f"sallm state database not found: {state_db}")
    convs = read_sallm_conversations(state_db, prefixes)
    if limit is not None:
        # Most recently active first when limiting.
        convs = sorted(convs, key=lambda c: c.updated_at or datetime.min, reverse=True)[:limit]
    return import_conversations(cfg.db_dsn, convs, folder=folder, tags=tags, dry_run=dry_run)
