"""HTTP routes for the Pawn portal (``/v1/portal/*``). See docs/PORTAL.md.

Single-user: every route uses the shared API bearer token. Chats, folders,
clips, annotations and the workspace layout live in Postgres
(``portal_models``); files are the S3 vault through :class:`VaultStore`.

Collections are small (one user), so tag filters, tag trees and sorting run
in Python over the selected rows instead of JSON-specific SQL, which keeps the
same code on Postgres and the SQLite test database.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import uuid
from datetime import datetime, timezone
from typing import Any, AsyncIterator, Callable, Iterable, Optional

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import Response, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Text, cast, delete, func, or_, select, update
from sqlalchemy.orm import Session

from pawn_core.database import _get_session
from pawn_core.vault import (
    VaultConflict,
    VaultError,
    VaultNotFound,
    VaultStat,
    VaultStore,
    VaultWriteDenied,
    dump_frontmatter,
    is_obsidian_meta,
    normalize_vault_key,
    parse_frontmatter,
)
from pawn_server.core import file_index
from pawn_server.core.portal_models import (
    PortalAnnotation,
    PortalChat,
    PortalChatFolder,
    PortalClip,
    PortalFileIndex,
    PortalFileTags,
    PortalMessage,
    PortalWorkspace,
)

logger = logging.getLogger(__name__)

CHAT_SORTS = ("updated", "created", "name", "tag")
TARGET_KINDS = ("message", "file", "item", "session", "clip", "capture", "event")
_TITLE_LEN = 80
_TEXT_SUFFIXES = (".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".base", ".canvas")


# ── helpers ───────────────────────────────────────────────────────────────────


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _iso(value: Optional[datetime]) -> Optional[str]:
    return value.isoformat() + "Z" if value else None


def _new_id() -> str:
    return uuid.uuid4().hex


def clean_tags(tags: Optional[Iterable[str]]) -> list[str]:
    """Normalize hierarchical tags: trim, collapse slashes, drop ``#``, dedupe."""
    out: list[str] = []
    for raw in tags or []:
        parts = [p.strip() for p in str(raw).strip().lstrip("#").split("/")]
        tag = "/".join(p for p in parts if p)
        if tag and tag not in out:
            out.append(tag)
    return out


def tag_matches(tags: Optional[list[str]], prefix: str) -> bool:
    """True when any tag equals *prefix* or sits below it in the hierarchy."""
    want = "/".join(clean_tags([prefix]))
    return any(t == want or t.startswith(want + "/") for t in tags or [])


def build_tag_tree(tag_lists: Iterable[Optional[list[str]]]) -> list[dict[str, Any]]:
    """Tree of ``{name, path, count, children}``; a node counts its subtree."""
    root: dict[str, Any] = {"children": {}}
    for tags in tag_lists:
        seen: set[str] = set()
        for tag in tags or []:
            node = root
            path: list[str] = []
            for part in tag.split("/"):
                path.append(part)
                key = "/".join(path)
                child = node["children"].setdefault(
                    part, {"name": part, "path": key, "count": 0, "children": {}}
                )
                if key not in seen:
                    child["count"] += 1
                    seen.add(key)
                node = child

    def _out(node: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {**child, "children": _out(child)}
            for _, child in sorted(node["children"].items(), key=lambda kv: kv[0].lower())
        ]

    return _out(root)


def rename_tag(tags: Optional[list[str]], src: str, dst: str) -> Optional[list[str]]:
    """Rewrite the *src* prefix to *dst*; returns ``None`` when nothing changed."""
    changed = False
    out: list[str] = []
    for tag in tags or []:
        if tag == src or tag.startswith(src + "/"):
            tag = dst + tag[len(src) :]
            changed = True
        if tag not in out:
            out.append(tag)
    return out if changed else None


def sort_chats(chats: list[PortalChat], sort: str, order: str) -> list[PortalChat]:
    reverse = order == "desc"
    if sort == "name":
        return sorted(chats, key=lambda c: (c.title or "").lower(), reverse=reverse)
    if sort == "created":
        return sorted(chats, key=lambda c: c.created_at or datetime.min, reverse=reverse)
    if sort == "tag":
        # Untagged chats always sort last; within a tag, by name.
        tagged = [c for c in chats if c.tags]
        untagged = [c for c in chats if not c.tags]
        tagged.sort(
            key=lambda c: (sorted(c.tags or [])[0].lower(), (c.title or "").lower()),
            reverse=reverse,
        )
        untagged.sort(key=lambda c: (c.title or "").lower())
        return tagged + untagged
    return sorted(
        chats,
        key=lambda c: c.last_message_at or c.updated_at or c.created_at or datetime.min,
        reverse=reverse,
    )


def _chat_out(chat: PortalChat, *, message_count: Optional[int] = None) -> dict[str, Any]:
    out = {
        "id": chat.id,
        "conversation_key": chat.conversation_key,
        "title": chat.title,
        "folder_id": chat.folder_id,
        "position": chat.position,
        "tags": list(chat.tags or []),
        "pinned": chat.pinned,
        "archived": chat.archived,
        "parent_chat_id": chat.parent_chat_id,
        "fork_from_message_id": chat.fork_from_message_id,
        "origin": chat.origin,
        "title_auto": bool(chat.title_auto),
        "tags_auto": bool(chat.tags_auto),
        "last_message_at": _iso(chat.last_message_at),
        "created_at": _iso(chat.created_at),
        "updated_at": _iso(chat.updated_at),
    }
    if message_count is not None:
        out["message_count"] = message_count
    return out


def _message_out(msg: PortalMessage) -> dict[str, Any]:
    return {
        "id": msg.id,
        "chat_id": msg.chat_id,
        "seq": msg.seq,
        "role": msg.role,
        "content": msg.content,
        "context": msg.context or {},
        "run_id": msg.run_id,
        "model": msg.model,
        "created_at": _iso(msg.created_at),
    }


def _folder_out(folder: PortalChatFolder) -> dict[str, Any]:
    return {
        "id": folder.id,
        "parent_id": folder.parent_id,
        "name": folder.name,
        "position": folder.position,
        "created_at": _iso(folder.created_at),
        "updated_at": _iso(folder.updated_at),
    }


def _clip_out(clip: PortalClip) -> dict[str, Any]:
    return {
        "id": clip.id,
        "chat_id": clip.chat_id,
        "message_ids": list(clip.message_ids or []),
        "title": clip.title,
        "tags": list(clip.tags or []),
        "content_md": clip.content_md,
        "vault_key": clip.vault_key,
        "created_at": _iso(clip.created_at),
        "updated_at": _iso(clip.updated_at),
    }


def _annotation_out(ann: PortalAnnotation) -> dict[str, Any]:
    return {
        "id": ann.id,
        "target_kind": ann.target_kind,
        "target_ref": ann.target_ref,
        "anchor": ann.anchor,
        "body_md": ann.body_md,
        "tags": list(ann.tags or []),
        "color": ann.color,
        "side_chat_id": ann.side_chat_id,
        "created_at": _iso(ann.created_at),
        "updated_at": _iso(ann.updated_at),
    }


def _annotation_context(db: Session, anns: list[PortalAnnotation]) -> dict[str, dict[str, Any]]:
    """Where each annotation lives, for search hits: the chat of a message."""
    msg_ids = {a.target_ref for a in anns if a.target_kind == "message"}
    clip_ids = {a.target_ref for a in anns if a.target_kind == "clip"}
    msgs = (
        {m.id: m for m in db.scalars(select(PortalMessage).where(PortalMessage.id.in_(msg_ids)))}
        if msg_ids
        else {}
    )
    chats = (
        {
            c.id: c
            for c in db.scalars(
                select(PortalChat).where(PortalChat.id.in_({m.chat_id for m in msgs.values()}))
            )
        }
        if msgs
        else {}
    )
    clips = (
        {c.id: c for c in db.scalars(select(PortalClip).where(PortalClip.id.in_(clip_ids)))}
        if clip_ids
        else {}
    )
    out: dict[str, dict[str, Any]] = {}
    for a in anns:
        if a.target_kind == "message" and a.target_ref in msgs:
            chat = chats.get(msgs[a.target_ref].chat_id)
            out[a.id] = {
                "chat_id": msgs[a.target_ref].chat_id,
                "chat_title": chat.title if chat else "",
                "label": (chat.title if chat else "") or "Chat",
            }
        elif a.target_kind == "clip" and a.target_ref in clips:
            out[a.id] = {"label": clips[a.target_ref].title or "Clip"}
        elif a.target_kind == "file":
            out[a.id] = {"label": a.target_ref.rsplit("/", 1)[-1]}
    return out


def new_chat(
    db: Session,
    *,
    title: str = "",
    folder_id: Optional[str] = None,
    tags: Optional[Iterable[str]] = None,
    parent_chat_id: Optional[str] = None,
    origin: Optional[dict[str, Any]] = None,
) -> PortalChat:
    """Insert a portal chat (``conversation_key`` ``chat:{id}``) and flush it."""
    now = _now()
    chat_id = _new_id()
    tag_list = clean_tags(tags or [])
    chat = PortalChat(
        id=chat_id,
        conversation_key=f"chat:{chat_id}",
        title=(title or "").strip(),
        folder_id=folder_id,
        tags=tag_list,
        parent_chat_id=parent_chat_id,
        origin=origin,
        title_auto=not (title or "").strip(),
        tags_auto=not tag_list,
        created_at=now,
        updated_at=now,
    )
    db.add(chat)
    db.flush()
    return chat


def _get_chat(db: Session, chat_id: str) -> PortalChat:
    chat = db.get(PortalChat, chat_id)
    if chat is None:
        raise HTTPException(status_code=404, detail="Chat not found")
    return chat


def _messages(db: Session, chat_id: str) -> list[PortalMessage]:
    return list(
        db.scalars(
            select(PortalMessage)
            .where(PortalMessage.chat_id == chat_id)
            .order_by(PortalMessage.seq)
        )
    )


def _title_from(text: str) -> str:
    line = next((ln.strip() for ln in (text or "").splitlines() if ln.strip()), "")
    return line[:_TITLE_LEN].rstrip() + ("…" if len(line) > _TITLE_LEN else "")


def _folder_descendants(db: Session, folder_id: str) -> set[str]:
    folders = list(db.scalars(select(PortalChatFolder)))
    children: dict[Optional[str], list[str]] = {}
    for f in folders:
        children.setdefault(f.parent_id, []).append(f.id)
    out: set[str] = set()
    stack = [folder_id]
    while stack:
        cur = stack.pop()
        for child in children.get(cur, []):
            if child not in out:
                out.add(child)
                stack.append(child)
    return out


def scratch_dir(cfg: Any) -> str:
    portal = getattr(cfg, "portal", None)
    configured = (getattr(portal, "scratch_dir", "") or "").strip()
    if configured:
        root = getattr(getattr(cfg, "vault", None), "agent_root", "Pawn") or "Pawn"
        return normalize_vault_key(configured.replace("{agent_root}", root))
    root = getattr(getattr(cfg, "vault", None), "agent_root", "Pawn") or "Pawn"
    return f"{normalize_vault_key(root)}/Scratch"


def _store(cfg: Any) -> VaultStore:
    from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

    try:
        return vault_store_from_config(cfg)
    except Exception as exc:  # noqa: BLE001 - surfaced as 503
        raise HTTPException(status_code=503, detail=f"Vault is not configured: {exc}") from exc


def _vault_http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, VaultNotFound):
        return HTTPException(status_code=404, detail=f"Not found: {exc}")
    if isinstance(exc, VaultWriteDenied):
        return HTTPException(status_code=403, detail=str(exc))
    if isinstance(exc, VaultConflict):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=502, detail=str(exc))


def _visible_key(key: str) -> bool:
    return not is_obsidian_meta(key) and not key.startswith(".") and "/." not in key


def list_folder(
    keys: Iterable[str | VaultStat],
    prefix: str,
    tags: Optional[dict[str, list[str]]] = None,
) -> dict[str, Any]:
    """One level of a vault listing: direct files plus synthesized subfolders.

    With :class:`VaultStat` entries, files carry ``modified``/``size`` and
    folders the newest ``modified`` below them.
    """
    base = normalize_vault_key(prefix)
    base = base + "/" if base else ""
    folders: dict[str, dict[str, Any]] = {}
    files: list[tuple[str, Optional[VaultStat]]] = []
    for entry in keys:
        stat = entry if isinstance(entry, VaultStat) else None
        key = stat.key if stat else str(entry)
        if not key.startswith(base) or not _visible_key(key):
            continue
        rest = key[len(base) :]
        modified = _naive(stat.last_modified) if stat else None
        if "/" in rest:
            name = rest.split("/", 1)[0]
            folder = folders.setdefault(name, {"count": 0, "modified": None})
            folder["count"] += 1
            if modified and (folder["modified"] is None or modified > folder["modified"]):
                folder["modified"] = modified
        else:
            files.append((key, stat))
    tags = tags or {}
    return {
        "prefix": base.rstrip("/"),
        "folders": [
            {
                "name": name,
                "path": base + name,
                "count": info["count"],
                "modified": _iso(info["modified"]),
            }
            for name, info in sorted(folders.items(), key=lambda kv: kv[0].lower())
        ],
        "files": [
            {
                "name": key[len(base) :],
                "path": key,
                "modified": _iso(_naive(stat.last_modified)) if stat else None,
                "size": stat.size if stat else None,
                "tags": tags.get(key, []),
            }
            for key, stat in sorted(files, key=lambda kv: kv[0].lower())
        ],
    }


def _naive(value: Optional[datetime]) -> Optional[datetime]:
    if value is None:
        return None
    if value.tzinfo is not None:
        value = value.astimezone(timezone.utc).replace(tzinfo=None)
    return value


def file_tags_map(db: Session, keys: Iterable[str]) -> dict[str, list[str]]:
    """Effective tags per key: indexed (frontmatter ∪ stored), else stored."""
    wanted = list(keys)
    if not wanted:
        return {}
    out: dict[str, list[str]] = {}
    for key, tags in db.execute(
        select(PortalFileIndex.key, PortalFileIndex.tags).where(PortalFileIndex.key.in_(wanted))
    ).all():
        if tags:
            out[key] = list(tags)
    for key, tags in db.execute(
        select(PortalFileTags.key, PortalFileTags.tags).where(PortalFileTags.key.in_(wanted))
    ).all():
        merged = out.setdefault(key, [])
        for t in tags or []:
            if t not in merged:
                merged.append(t)
    return out


def _tagged_file_keys(db: Session) -> list[str]:
    keys = {k for k, tags in db.execute(select(PortalFileIndex.key, PortalFileIndex.tags)) if tags}
    keys.update(db.scalars(select(PortalFileTags.key)))
    return sorted(keys)


def split_query(q: str) -> tuple[str, list[str]]:
    """Split ``#tag`` tokens out of a search query: ``(text, tags)``."""
    words: list[str] = []
    tags: list[str] = []
    for token in (q or "").split():
        if token.startswith("#") and len(token) > 1:
            tags.extend(clean_tags([token]))
        else:
            words.append(token)
    return " ".join(words).strip(), tags


def _file_match(db: Session, text: str, like: str) -> Any:
    """WHERE clause for file search: names/titles by LIKE, bodies by FTS on Postgres."""
    by_name = or_(PortalFileIndex.key.ilike(like), PortalFileIndex.title.ilike(like))
    bind = db.get_bind()
    if bind is not None and bind.dialect.name == "postgresql":
        vector = func.to_tsvector("simple", PortalFileIndex.title + " " + PortalFileIndex.content)
        query = func.websearch_to_tsquery("simple", text)
        return or_(by_name, vector.op("@@")(query), PortalFileIndex.content.ilike(like))
    return or_(by_name, PortalFileIndex.content.ilike(like))


def _rewrite_note_tags(store: VaultStore, key: str, src: str, dst: str) -> bool:
    """Rename a tag inside a writable note's frontmatter; True when written."""
    stat = store.stat(key)
    if stat is None:
        return False
    text = store.read(key)
    store.assert_writable(key, existing_body=text)
    meta, body = parse_frontmatter(text)
    new = rename_tag(file_index.frontmatter_tags(meta), src, dst)
    if new is None:
        return False
    meta.pop("tag", None)
    meta["tags"] = new
    store.write_if_unchanged(key, dump_frontmatter(meta, body), stat.etag)
    return True


def _move_file_tags(cfg: Any, src: str, dst: str) -> None:
    with _get_session(cfg.db_dsn) as db:
        row = db.get(PortalFileTags, src)
        if row is None:
            return
        tags = list(row.tags or [])
        db.delete(row)
        db.flush()
        target = db.get(PortalFileTags, dst) or PortalFileTags(key=dst)
        target.tags = tags
        target.updated_at = _now()
        db.merge(target)


def _drop_file_rows(cfg: Any, key: str) -> None:
    with _get_session(cfg.db_dsn) as db:
        db.execute(delete(PortalFileTags).where(PortalFileTags.key == key))
        db.execute(delete(PortalFileIndex).where(PortalFileIndex.key == key))


def _file_tags_state(cfg: Any, store: VaultStore, key: str) -> dict[str, Any]:
    """``{key, tags, frontmatter, stored, source}`` for one vault file."""
    fm: list[str] = []
    writable = False
    if key.endswith(".md"):
        try:
            text = store.read(key)
        except VaultNotFound as exc:
            raise HTTPException(status_code=404, detail="Not found") from exc
        fm = file_index.frontmatter_tags(parse_frontmatter(text)[0])
        try:
            store.assert_writable(key, existing_body=text)
            writable = True
        except VaultWriteDenied:
            writable = False
    with _get_session(cfg.db_dsn) as db:
        row = db.get(PortalFileTags, key)
        stored = list(row.tags or []) if row else []
    tags = list(fm) + [t for t in stored if t not in fm]
    return {
        "key": key,
        "tags": tags,
        "frontmatter": fm,
        "stored": stored,
        "source": "frontmatter" if writable else "db",
    }


def _set_file_tags(cfg: Any, store: VaultStore, key: str, tags: list[str]) -> dict[str, Any]:
    """Frontmatter for writable notes, ``portal_file_tags`` for everything else."""
    if key.endswith(".md"):
        stat = store.stat(key)
        if stat is None:
            raise VaultNotFound(key)
        text = store.read(key)
        try:
            store.assert_writable(key, existing_body=text)
            writable = True
        except VaultWriteDenied:
            writable = False
        meta, body = parse_frontmatter(text)
        if writable:
            meta.pop("tag", None)
            if tags:
                meta["tags"] = tags
            else:
                meta.pop("tags", None)
            store.write_if_unchanged(key, dump_frontmatter(meta, body), stat.etag)
            with _get_session(cfg.db_dsn) as db:
                db.execute(delete(PortalFileTags).where(PortalFileTags.key == key))
        else:
            fm = file_index.frontmatter_tags(meta)
            _store_file_tags(cfg, key, [t for t in tags if t not in fm])
        file_index.index_keys(cfg, [key], store)
    else:
        if store.stat(key) is None:
            raise VaultNotFound(key)
        _store_file_tags(cfg, key, tags)
        file_index.refresh_tags(cfg, key, [])
    return _file_tags_state(cfg, store, key)


def _store_file_tags(cfg: Any, key: str, tags: list[str]) -> None:
    with _get_session(cfg.db_dsn) as db:
        row = db.get(PortalFileTags, key)
        if not tags:
            if row is not None:
                db.delete(row)
            return
        if row is None:
            row = PortalFileTags(key=key)
            db.add(row)
        row.tags = tags
        row.updated_at = _now()


# ── request bodies ────────────────────────────────────────────────────────────


class ChatCreate(BaseModel):
    title: str = ""
    folder_id: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    parent_chat_id: Optional[str] = None
    origin: Optional[dict[str, Any]] = None


class ChatPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: Optional[str] = None
    folder_id: Optional[str] = None
    position: Optional[int] = None
    tags: Optional[list[str]] = None
    pinned: Optional[bool] = None
    archived: Optional[bool] = None
    # Explicit "move to Unfiled" (folder_id null is otherwise "unchanged").
    unfile: bool = False
    # Side chats: promoting keeps ``origin`` as a link but drops the badge and
    # the nesting under the parent (``origin.promoted``).
    promoted: Optional[bool] = None


class ChatsMove(BaseModel):
    ids: list[str]
    folder_id: Optional[str] = None


class FolderCreate(BaseModel):
    name: str
    parent_id: Optional[str] = None
    position: int = 0


class FolderPatch(BaseModel):
    name: Optional[str] = None
    parent_id: Optional[str] = None
    position: Optional[int] = None
    to_root: bool = False


class TagRename(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    src: str = Field(alias="from")
    dst: str = Field(alias="to")


class PortalContextNote(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str
    content: Optional[str] = None
    inline: bool = True


class PortalImage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    filename: str
    media_type: str = ""
    data_base64: str
    role: str = "question"


class MessageCreate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    message: str = ""
    active_note: Optional[PortalContextNote] = None
    selection: Optional[str] = None
    context: list[PortalContextNote] = Field(default_factory=list)
    images: list[PortalImage] = Field(default_factory=list)
    model: Optional[str] = None
    reasoning: Optional[str] = None
    route: Optional[str] = None
    # Board chats: the cards selected on the board when the message was sent.
    board_selection: list[str] = Field(default_factory=list)


class ForkRequest(BaseModel):
    message_id: str
    title: Optional[str] = None


class ClipCreate(BaseModel):
    chat_id: Optional[str] = None
    message_ids: list[str] = Field(default_factory=list)
    title: str = ""
    tags: list[str] = Field(default_factory=list)
    # Optional explicit body (a selection); otherwise built from the messages.
    content_md: Optional[str] = None


class ClipPatch(BaseModel):
    title: Optional[str] = None
    tags: Optional[list[str]] = None
    content_md: Optional[str] = None


class ClipSave(BaseModel):
    path: Optional[str] = None


class AnnotationCreate(BaseModel):
    target_kind: str
    target_ref: str
    anchor: Optional[dict[str, Any]] = None
    body_md: str = ""
    tags: list[str] = Field(default_factory=list)
    color: Optional[str] = None
    side_chat_id: Optional[str] = None


class AnnotationPatch(BaseModel):
    anchor: Optional[dict[str, Any]] = None
    body_md: Optional[str] = None
    tags: Optional[list[str]] = None
    color: Optional[str] = None
    side_chat_id: Optional[str] = None


class FileWrite(BaseModel):
    key: str
    content: str
    # Empty = create (must not exist); otherwise the etag last read.
    etag: str = ""


class FileTagsPut(BaseModel):
    key: str
    tags: list[str] = Field(default_factory=list)


class FileMove(BaseModel):
    src: str
    dst: str
    overwrite: bool = False


class WorkspacePut(BaseModel):
    layout: dict[str, Any] = Field(default_factory=dict)
    name: str = "default"


# ── portal mirror hooks for the shared chat stream ────────────────────────────


def _item_context(cfg: Any, item_id: str) -> tuple[Optional[str], list[dict]]:
    """A coworker item as side-chat grounding: its text, quote and reason."""
    try:
        from pawn_agent.core.coworker import db as itemdb  # noqa: PLC0415

        item = itemdb.get_item(cfg.db_dsn, item_id)
    except Exception as exc:  # noqa: BLE001 - grounding is best effort
        logger.debug("item origin lookup failed: %s", exc)
        return None, []
    if not item:
        return None, []
    parts = [
        f"Coworker item ({item.get('kind') or 'item'}, {item.get('status') or 'new'}): {item.get('text') or ''}"
    ]
    if item.get("quote"):
        parts.append(f"Quote: {item['quote']}")
    if item.get("reason"):
        parts.append(f"Why it surfaced: {item['reason']}")
    if item.get("source_kind") or item.get("source_ref"):
        parts.append(
            f"Source: {item.get('source_kind') or ''} {item.get('source_ref') or ''}".strip()
        )
    notes = [{"path": str(item["note_key"]), "inline": True}] if item.get("note_key") else []
    return "\n".join(parts), notes


def _merge_quote(quote: Optional[str], extra: Optional[str]) -> Optional[str]:
    parts = [p.strip() for p in (quote, extra) if p and str(p).strip()]
    return "\n\n".join(parts) or None


def _clock(seconds: float) -> str:
    """Name a transcript offset as ``m:ss`` or ``h:mm:ss``."""
    total = max(0, int(round(float(seconds))))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes}:{secs:02d}"


def _read_note(cfg: Any, path: str) -> Optional[str]:
    try:
        return _store(cfg).read(path)
    except Exception as exc:  # noqa: BLE001 - grounding is best effort
        logger.debug("origin note %s unavailable: %s", path, exc)
        return None


def _session_origin(cfg: Any, origin: dict[str, Any]) -> tuple[Optional[str], list[dict]]:
    """Transcript note when mapped; otherwise a header plus the transcript tool hint."""
    sid = str(origin.get("ref") or "").strip()
    detail: dict[str, Any] = {
        "session_id": sid,
        "title": sid,
        "speakers": [],
        "transcript_path": None,
    }
    try:
        from pawn_server.core.sessions_api import get_session  # noqa: PLC0415

        detail = get_session(cfg, sid)
    except Exception as exc:  # noqa: BLE001 - grounding is best effort
        logger.debug("session origin lookup failed: %s", exc)

    title = str(detail.get("title") or sid)
    when = str(detail.get("created_at") or detail.get("updated_at") or "")
    speakers = [str(s) for s in (detail.get("speakers") or []) if str(s).strip()]
    path = str(detail.get("transcript_path") or "").strip()
    lines = [f"Session: {title}"]
    if when:
        lines.append(f"Date: {when}")
    if path:
        lines.append(f"Speakers: {', '.join(speakers)}" if speakers else "Speakers: unknown")
    else:
        lines.append(f"Speakers: {', '.join(speakers) if speakers else 'unknown'}")
    lines.append(f"Session id: {sid}")
    notes: list[dict] = []
    if path:
        notes.append({"path": path, "inline": True})
    else:
        lines.append(f"use `session_transcript --session-id {sid}`")
    raw_t = origin.get("t")
    if raw_t is not None and str(raw_t).strip() != "":
        try:
            secs = float(raw_t)
            lines.append(f"Timestamp: {_clock(secs)} ({secs:g}s)")
        except (TypeError, ValueError):
            lines.append(f"Timestamp: {raw_t}")
    return "\n".join(lines), notes


def _event_origin(cfg: Any, origin: dict[str, Any]) -> tuple[Optional[str], list[dict]]:
    """Inline the event note (its body includes ``## Prep``)."""
    eid = str(origin.get("ref") or "").strip()
    try:
        from pawn_server.core.planner import find_event_by_id  # noqa: PLC0415

        ev = find_event_by_id(cfg, eid)
    except Exception as exc:  # noqa: BLE001 - grounding is best effort
        logger.debug("event origin lookup failed: %s", exc)
        return f"Event: {eid}", []
    path = str(ev.get("path") or "").strip()
    notes = [{"path": path, "inline": True}] if path else []
    title = str(ev.get("title") or eid)
    start = str(ev.get("start") or "")
    line = f"Event: {title}" + (f" ({start})" if start else "")
    return line, notes


def _render_day(day: dict[str, Any]) -> str:
    """Compact text of one planner day, for a side chat's first turn."""
    lines = [f"Day {day.get('date') or ''}"]
    events = list(day.get("events") or [])
    if events:
        lines.append("Events:")
        for ev in events:
            if not isinstance(ev, dict):
                continue
            start = str(ev.get("start") or "")
            end = str(ev.get("end") or "")
            clock = ""
            if "T" in start:
                clock = start.split("T", 1)[1][:5]
                if "T" in end:
                    clock += "–" + end.split("T", 1)[1][:5]
            title = str(ev.get("title") or "Untitled")
            bits = [b for b in (clock, title) if b]
            attendees = [str(a) for a in (ev.get("attendees") or []) if str(a).strip()]
            if attendees:
                bits.append("(" + ", ".join(attendees[:6]) + ")")
            lines.append("- " + " ".join(bits))
    else:
        lines.append("Events: none")
    sessions = [s for s in (day.get("sessions") or []) if isinstance(s, dict)]
    if sessions:
        lines.append("Sessions:")
        for sess in sessions:
            dur = sess.get("duration_seconds")
            dur_s = ""
            try:
                if dur is not None:
                    dur_s = f" ({int(float(dur) // 60)}m)"
            except (TypeError, ValueError):
                dur_s = ""
            lines.append(f"- {sess.get('title') or sess.get('session_id')}{dur_s}")
    others = day.get("other_items") or []
    lines.append(f"Open TODOs: {len(others) if isinstance(others, list) else 0}")
    lines.append("Use the day_plan skill.")
    return "\n".join(lines)


def _day_origin(cfg: Any, origin: dict[str, Any]) -> tuple[Optional[str], list[dict]]:
    ref = str(origin.get("ref") or "").strip()
    try:
        from pawn_server.core.planner import build_day  # noqa: PLC0415

        payload = build_day(cfg, ref)
    except Exception as exc:  # noqa: BLE001 - grounding is best effort
        logger.debug("day origin lookup failed: %s", exc)
        payload = {"date": ref, "events": [], "sessions": [], "other_items": []}
    return _render_day(payload), []


def _people_note_path(cfg: Any, speaker_id: str) -> str:
    try:
        from pawn_agent.core.people.notes import people_note_key  # noqa: PLC0415

        return people_note_key(cfg, speaker_id)
    except Exception as exc:  # noqa: BLE001
        logger.debug("people note path failed: %s", exc)
        return f"People/{speaker_id}.md"


def _speaker_summary(cfg: Any, speaker_id: str) -> str:
    lines = [f"Speaker id: {speaker_id}"]
    try:
        from pawn_diarize.core.speaker_gallery import SpeakerGallery  # noqa: PLC0415

        gallery = SpeakerGallery(cfg.db_dsn, config=getattr(cfg, "speakers", None))
        speaker = gallery.get_speaker(speaker_id) or gallery.find_speaker_by_name(speaker_id)
        if speaker is None:
            return "\n".join(lines)
        enrollments = gallery.list_enrollments(speaker.id)
        seconds = sum(float(getattr(row, "duration", 0) or 0) for row in enrollments)
        aliases = ", ".join(str(a) for a in (speaker.aliases or []) if str(a).strip()) or "none"
        lines = [
            f"Speaker: {speaker.display_name} ({speaker.id})",
            f"Aliases: {aliases}",
            f"Enrollments: {len(enrollments)} ({int(seconds)}s)",
            f"Active: {'yes' if speaker.active else 'no'}",
        ]
        if getattr(speaker, "notes", None):
            lines.append(f"Notes: {speaker.notes}")
    except Exception as exc:  # noqa: BLE001 - grounding is best effort
        logger.debug("speaker origin lookup failed: %s", exc)
    return "\n".join(lines)


def _speaker_origin(cfg: Any, origin: dict[str, Any]) -> tuple[Optional[str], list[dict]]:
    """People note when it exists, plus the gallery card."""
    sid = str(origin.get("ref") or "").strip()
    path = _people_note_path(cfg, sid)
    notes: list[dict] = []
    if _read_note(cfg, path) is not None:
        notes.append({"path": path, "inline": True})
    return _speaker_summary(cfg, sid), notes


def _origin_context(
    origin: Optional[dict[str, Any]], cfg: Any = None
) -> tuple[Optional[str], list[dict]]:
    """Selection text + context notes that seed a side chat's first prompt."""
    if not origin:
        return None, []
    quote = (origin.get("quote") or "").strip() or None
    notes: list[dict] = []
    kind = origin.get("kind")
    if kind == "file" and origin.get("ref"):
        notes.append({"path": str(origin["ref"]), "inline": True})
    if kind == "item" and origin.get("ref") and cfg is not None:
        item_text, item_notes = _item_context(cfg, str(origin["ref"]))
        if item_text:
            quote = _merge_quote(quote, item_text)
        notes.extend(item_notes)
    if cfg is not None and origin.get("ref"):
        extra: Optional[str] = None
        extra_notes: list[dict] = []
        if kind == "session":
            extra, extra_notes = _session_origin(cfg, origin)
        elif kind == "event":
            extra, extra_notes = _event_origin(cfg, origin)
        elif kind == "day":
            extra, extra_notes = _day_origin(cfg, origin)
        elif kind == "speaker":
            extra, extra_notes = _speaker_origin(cfg, origin)
        if extra:
            quote = _merge_quote(quote, extra)
        notes.extend(extra_notes)
    return quote, notes


class PortalMirrorHooks:
    """Persist the user turn and the answer as ``portal_messages`` rows."""

    def __init__(self, cfg: Any, chat_id: str, body: MessageCreate, context: dict) -> None:
        self.cfg = cfg
        self.chat_id = chat_id
        self.body = body
        self.context = context

    def _append(self, role: str, content: str, **extra: Any) -> PortalMessage:
        with _get_session(self.cfg.db_dsn) as db:
            chat = _get_chat(db, self.chat_id)
            last = db.scalar(
                select(func.max(PortalMessage.seq)).where(PortalMessage.chat_id == self.chat_id)
            )
            now = _now()
            msg = PortalMessage(
                id=_new_id(),
                chat_id=self.chat_id,
                seq=(last + 1) if last is not None else 0,
                role=role,
                content=content,
                created_at=now,
                **extra,
            )
            db.add(msg)
            chat.last_message_at = now
            chat.updated_at = now
            if role == "user" and not (chat.title or "").strip():
                chat.title = _title_from(content) or "New chat"
            db.flush()
            db.expunge(msg)
            return msg

    async def on_user(self, message: str) -> list[tuple[str, dict]]:
        msg = await asyncio.to_thread(
            self._append,
            "user",
            message,
            context=self.context or None,
            model=(self.body.model or "").strip() or None,
        )
        return [("message", {"id": msg.id, "seq": msg.seq, "role": "user"})]

    async def on_answer(
        self,
        content: str,
        *,
        run_id: Optional[str],
        written_files: list[str],
        extra_context: Optional[dict] = None,
    ) -> list[tuple[str, dict]]:
        ctx: Optional[dict] = {"written_files": written_files} if written_files else None
        if extra_context:
            ctx = {**(ctx or {}), **extra_context}
        msg = await asyncio.to_thread(
            self._append,
            "assistant",
            content,
            context=ctx,
            run_id=run_id,
            model=(self.body.model or "").strip() or None,
        )
        events: list[tuple[str, dict]] = [
            ("message", {"id": msg.id, "seq": msg.seq, "role": "assistant"})
        ]
        try:
            update = await asyncio.wait_for(
                auto_name_chat(self.cfg, self.chat_id), _AUTONAME_TIMEOUT
            )
        except Exception as exc:  # noqa: BLE001 - naming is best effort
            logger.info("portal auto-name skipped for %s: %s", self.chat_id, exc)
            update = None
        if update:
            events.append(("chat", update))
        return events


_AUTONAME_TIMEOUT = 20.0
_AUTONAME_CHARS = 6000
_AUTONAME_PROMPT = """You name and tag saved chats in a personal knowledge workspace.
Read the conversation and reply with ONLY a JSON object (no fences):
{{"title": "...", "tags": ["...", "..."]}}
- title: 3-8 words, specific to the topic, no quotes, no trailing period.
- tags: 1-4 lowercase tags; use "/" for hierarchy (e.g. "work/security").
  Prefer reusing these existing tags when they fit: {vocabulary}
Conversation:
{transcript}"""


def _autoname_due(chat: PortalChat, last_seq: int, every: int) -> bool:
    if not (chat.title_auto or chat.tags_auto):
        return False
    if chat.autoname_seq is None:
        return True
    return every > 0 and last_seq - chat.autoname_seq >= every


def _autoname_transcript(msgs: list[PortalMessage]) -> str:
    lines = [f"{'User' if m.role == 'user' else 'Assistant'}: {m.content.strip()}" for m in msgs]
    text = "\n\n".join(lines)
    if len(text) <= _AUTONAME_CHARS:
        return text
    head = text[: _AUTONAME_CHARS * 2 // 3]
    tail = text[-_AUTONAME_CHARS // 3 :]
    return f"{head}\n\n[…]\n\n{tail}"


async def auto_name_chat(cfg: Any, chat_id: str) -> Optional[dict[str, Any]]:
    """Let the background model (re)name and tag a chat; returns the change.

    Runs after the first answer and again every ``portal.auto_name_every``
    messages, only for fields the user has not edited (``title_auto`` /
    ``tags_auto``).
    """
    portal = getattr(cfg, "portal", None)
    if not getattr(portal, "auto_name", True):
        return None
    every = int(getattr(portal, "auto_name_every", 6) or 0)

    def _load() -> Optional[tuple[str, list[str]]]:
        with _get_session(cfg.db_dsn) as db:
            chat = db.get(PortalChat, chat_id)
            msgs = _messages(db, chat_id)
            if chat is None or not msgs or not any(m.role == "user" for m in msgs):
                return None
            if not _autoname_due(chat, msgs[-1].seq, every):
                return None
            counts: dict[str, int] = {}
            for tags in db.scalars(select(PortalChat.tags)):
                for t in tags or []:
                    counts[t] = counts.get(t, 0) + 1
            vocab = sorted(counts, key=lambda t: (-counts[t], t))[:50]
            return _autoname_transcript(msgs), vocab

    loaded = await asyncio.to_thread(_load)
    if loaded is None:
        return None
    transcript, vocab = loaded
    from pawn_server.core.background_llm import (  # noqa: PLC0415
        complete_text,
        parse_json_object,
    )

    prompt = _AUTONAME_PROMPT.format(
        vocabulary=", ".join(vocab) if vocab else "(none yet)", transcript=transcript
    )
    raw = await asyncio.to_thread(
        complete_text, cfg, [{"role": "user", "content": prompt}], session="pawn-portal-autoname"
    )
    data = parse_json_object(raw)
    title = " ".join(str(data.get("title") or "").split()).strip(" .\"'")[:_TITLE_LEN]
    raw_tags = data.get("tags")
    tags = clean_tags([str(t).lower() for t in raw_tags][:4]) if isinstance(raw_tags, list) else []
    if not title and not tags:
        return None

    def _apply() -> Optional[dict[str, Any]]:
        with _get_session(cfg.db_dsn) as db:
            chat = db.get(PortalChat, chat_id)
            if chat is None:
                return None
            last = db.scalar(
                select(func.max(PortalMessage.seq)).where(PortalMessage.chat_id == chat_id)
            )
            chat.autoname_seq = int(last or 0)
            changed = False
            if chat.title_auto and title and title != chat.title:
                chat.title = title
                changed = True
            if chat.tags_auto and tags and tags != list(chat.tags or []):
                chat.tags = tags
                changed = True
            if not changed:
                return None
            chat.updated_at = _now()
            return {"id": chat.id, "title": chat.title, "tags": list(chat.tags or [])}

    return await asyncio.to_thread(_apply)


def _clip_markdown(chat: Optional[PortalChat], msgs: list[PortalMessage]) -> str:
    lines: list[str] = []
    for msg in msgs:
        who = "You" if msg.role == "user" else "Pawn"
        lines.append(f"**{who}:**\n\n{msg.content.strip()}\n")
    if chat is not None:
        lines.append(f"_From chat: {chat.title or chat.id}_")
    return "\n".join(lines).strip() + "\n"


def _safe_filename(title: str) -> str:
    bad = '\\/:*?"<>|#^[]'
    name = "".join("-" if ch in bad else ch for ch in title).strip(" .-")
    return name[:80] or "clip"


# ── router ────────────────────────────────────────────────────────────────────


def build_router(get_cfg: Callable[[], Any], require_token: Callable[..., None]) -> APIRouter:
    """Build the router; dependencies come from api_server to avoid a cycle."""
    router = APIRouter(prefix="/v1/portal", tags=["portal"], dependencies=[Depends(require_token)])

    # ── chats ────────────────────────────────────────────────────────────────

    @router.get("/chats")
    def chats_list(
        q: Optional[str] = None,
        tag: Optional[str] = None,
        folder_id: Optional[str] = None,
        unfiled: bool = False,
        pinned: Optional[bool] = None,
        archived: bool = False,
        origin_ref: Optional[str] = None,
        sort: str = Query(default="updated"),
        order: str = Query(default="desc", pattern="^(asc|desc)$"),
        cfg: Any = Depends(get_cfg),
    ) -> dict:
        if sort not in CHAT_SORTS:
            raise HTTPException(status_code=422, detail=f"sort must be one of {CHAT_SORTS}")
        with _get_session(cfg.db_dsn) as db:
            stmt = select(PortalChat).where(PortalChat.archived == archived)
            if folder_id:
                stmt = stmt.where(PortalChat.folder_id == folder_id)
            elif unfiled:
                stmt = stmt.where(PortalChat.folder_id.is_(None))
            if pinned is not None:
                stmt = stmt.where(PortalChat.pinned == pinned)
            if q:
                like = f"%{q.strip()}%"
                hits = select(PortalMessage.chat_id).where(PortalMessage.content.ilike(like))
                stmt = stmt.where(or_(PortalChat.title.ilike(like), PortalChat.id.in_(hits)))
            chats = list(db.scalars(stmt))
            if tag:
                chats = [c for c in chats if tag_matches(c.tags, tag)]
            if origin_ref:
                chats = [c for c in chats if (c.origin or {}).get("ref") == origin_ref]
            counts = dict(
                db.execute(
                    select(PortalMessage.chat_id, func.count())
                    .where(PortalMessage.chat_id.in_([c.id for c in chats]))
                    .group_by(PortalMessage.chat_id)
                ).all()
            )
            ordered = sort_chats(chats, sort, order)
            return {"chats": [_chat_out(c, message_count=counts.get(c.id, 0)) for c in ordered]}

    @router.post("/chats", status_code=201)
    def chats_create(body: ChatCreate, cfg: Any = Depends(get_cfg)) -> dict:
        with _get_session(cfg.db_dsn) as db:
            if body.folder_id and db.get(PortalChatFolder, body.folder_id) is None:
                raise HTTPException(status_code=404, detail="Folder not found")
            if body.parent_chat_id:
                _get_chat(db, body.parent_chat_id)
            chat = new_chat(
                db,
                title=body.title,
                folder_id=body.folder_id,
                tags=body.tags,
                parent_chat_id=body.parent_chat_id,
                origin=body.origin,
            )
            return _chat_out(chat, message_count=0)

    @router.get("/chats/{chat_id}")
    def chats_get(chat_id: str, cfg: Any = Depends(get_cfg)) -> dict:
        with _get_session(cfg.db_dsn) as db:
            chat = _get_chat(db, chat_id)
            msgs = _messages(db, chat_id)
            side = list(db.scalars(select(PortalChat).where(PortalChat.parent_chat_id == chat_id)))
            out = _chat_out(chat, message_count=len(msgs))
            out["messages"] = [_message_out(m) for m in msgs]
            out["side_chats"] = [_chat_out(c) for c in side]
            return out

    @router.patch("/chats/{chat_id}")
    def chats_patch(chat_id: str, body: ChatPatch, cfg: Any = Depends(get_cfg)) -> dict:
        with _get_session(cfg.db_dsn) as db:
            chat = _get_chat(db, chat_id)
            if body.title is not None:
                chat.title = body.title.strip()
                chat.title_auto = False
            if body.unfile:
                chat.folder_id = None
            elif body.folder_id is not None:
                if db.get(PortalChatFolder, body.folder_id) is None:
                    raise HTTPException(status_code=404, detail="Folder not found")
                chat.folder_id = body.folder_id
            if body.position is not None:
                chat.position = body.position
            if body.tags is not None:
                chat.tags = clean_tags(body.tags)
                chat.tags_auto = False
            if body.pinned is not None:
                chat.pinned = body.pinned
            if body.archived is not None:
                chat.archived = body.archived
            if body.promoted is not None:
                if not chat.origin:
                    raise HTTPException(status_code=422, detail="Not a side chat")
                chat.origin = {**chat.origin, "promoted": body.promoted}
            chat.updated_at = _now()
            db.flush()
            return _chat_out(chat)

    @router.delete("/chats/{chat_id}")
    async def chats_delete(chat_id: str, cfg: Any = Depends(get_cfg)) -> dict:
        from pawn_server.core.api_server import get_sallm_registry  # noqa: PLC0415

        def _drop() -> str:
            with _get_session(cfg.db_dsn) as db:
                chat = _get_chat(db, chat_id)
                key = chat.conversation_key
                db.execute(delete(PortalMessage).where(PortalMessage.chat_id == chat_id))
                db.execute(
                    update(PortalChat)
                    .where(PortalChat.parent_chat_id == chat_id)
                    .values(parent_chat_id=None)
                )
                # Annotations keep their note; only the side-chat link goes.
                db.execute(
                    update(PortalAnnotation)
                    .where(PortalAnnotation.side_chat_id == chat_id)
                    .values(side_chat_id=None)
                )
                db.delete(chat)
                return key

        key = await asyncio.to_thread(_drop)
        try:
            await get_sallm_registry().reset(key, cfg.db_dsn)
        except Exception:  # noqa: BLE001 - mirror is gone; sallm cleanup is best effort
            pass
        return {"deleted": chat_id}

    @router.post("/chats/move")
    def chats_move(body: ChatsMove, cfg: Any = Depends(get_cfg)) -> dict:
        with _get_session(cfg.db_dsn) as db:
            if body.folder_id and db.get(PortalChatFolder, body.folder_id) is None:
                raise HTTPException(status_code=404, detail="Folder not found")
            result = db.execute(
                update(PortalChat)
                .where(PortalChat.id.in_(body.ids))
                .values(folder_id=body.folder_id, updated_at=_now())
            )
            return {"moved": result.rowcount, "folder_id": body.folder_id}

    @router.post("/chats/{chat_id}/messages")
    async def chats_message(
        chat_id: str, body: MessageCreate, cfg: Any = Depends(get_cfg)
    ) -> StreamingResponse:
        from pawn_server.core.api_server import (  # noqa: PLC0415
            ChatImage,
            ContextNote,
            PawnChatRequest,
            pawn_chat_stream,
        )

        def _load() -> tuple[str, Optional[dict], int]:
            with _get_session(cfg.db_dsn) as db:
                chat = _get_chat(db, chat_id)
                count = db.scalar(select(func.count()).where(PortalMessage.chat_id == chat_id))
                return chat.conversation_key, chat.origin, int(count or 0)

        from pawn_server.core.boards import board_for_origin  # noqa: PLC0415

        conversation, origin, count = await asyncio.to_thread(_load)
        selection = body.selection
        notes = [n.model_dump() for n in body.context]
        board = board_for_origin(origin)
        if count == 0 and origin and not board:
            # First turn of a side chat: ground it in what it was spawned from.
            quote, origin_notes = await asyncio.to_thread(_origin_context, origin, cfg)
            selection = selection or quote
            notes = origin_notes + [
                n for n in notes if n["path"] not in {o["path"] for o in origin_notes}
            ]
        request = PawnChatRequest(
            conversation=conversation,
            message=body.message,
            active_note=ContextNote(**body.active_note.model_dump()) if body.active_note else None,
            selection=selection,
            context=[ContextNote(**n) for n in notes],
            images=[ChatImage(**i.model_dump()) for i in body.images],
            model=body.model,
            reasoning=body.reasoning,
            route=body.route,
        )
        context: dict[str, Any] = {}
        if selection:
            context["selection"] = selection
        if notes:
            context["notes"] = [n["path"] for n in notes]
        if body.active_note:
            context["active_note"] = body.active_note.path
        if body.images:
            context["images"] = [
                {"filename": i.filename, "media_type": i.media_type, "role": i.role}
                for i in body.images
            ]
        if count == 0 and origin:
            context["origin"] = origin
        if board and body.board_selection:
            context["board_selection"] = list(body.board_selection)
        hooks = PortalMirrorHooks(cfg, chat_id, body, context)
        stream: AsyncIterator[str] = pawn_chat_stream(
            request,
            cfg,
            hooks,
            source="portal",
            board=board,
            board_selection=list(body.board_selection),
        )
        return StreamingResponse(stream, media_type="text/event-stream")

    @router.post("/chats/{chat_id}/fork", status_code=201)
    async def chats_fork(chat_id: str, body: ForkRequest, cfg: Any = Depends(get_cfg)) -> dict:
        def _copy() -> tuple[dict, str, str, int, list[tuple[str, str]]]:
            with _get_session(cfg.db_dsn) as db:
                src = _get_chat(db, chat_id)
                msgs = _messages(db, chat_id)
                cut = next((m for m in msgs if m.id == body.message_id), None)
                if cut is None:
                    raise HTTPException(status_code=404, detail="Message not in this chat")
                kept = [m for m in msgs if m.seq <= cut.seq]
                now = _now()
                new_id = _new_id()
                chat = PortalChat(
                    id=new_id,
                    conversation_key=f"chat:{new_id}",
                    title=(body.title or "").strip() or f"{src.title or 'Chat'} (fork)",
                    folder_id=src.folder_id,
                    tags=list(src.tags or []),
                    parent_chat_id=src.id,
                    fork_from_message_id=cut.id,
                    last_message_at=kept[-1].created_at if kept else now,
                    created_at=now,
                    updated_at=now,
                )
                db.add(chat)
                db.flush()
                for i, m in enumerate(kept):
                    db.add(
                        PortalMessage(
                            id=_new_id(),
                            chat_id=new_id,
                            seq=i,
                            role=m.role,
                            content=m.content,
                            context=m.context,
                            run_id=m.run_id,
                            model=m.model,
                            created_at=m.created_at,
                        )
                    )
                db.flush()
                pairs = [(m.role, m.content) for m in kept if m.role in ("user", "assistant")]
                return (
                    _chat_out(chat, message_count=len(kept)),
                    src.conversation_key,
                    chat.conversation_key,
                    len(pairs),
                    pairs,
                )

        out, src_key, dst_key, turns, pairs = await asyncio.to_thread(_copy)
        try:
            from pawn_agent.core.sallm_fork import fork_session  # noqa: PLC0415

            seeded = await asyncio.to_thread(
                fork_session, cfg, src_key, dst_key, turns=turns, fallback=pairs
            )
        except Exception as exc:  # noqa: BLE001 - the mirror fork still works
            out["sallm_seeded"] = 0
            out["sallm_error"] = str(exc)
        else:
            out["sallm_seeded"] = seeded
        return out

    # ── folders ──────────────────────────────────────────────────────────────

    @router.get("/folders")
    def folders_list(cfg: Any = Depends(get_cfg)) -> dict:
        with _get_session(cfg.db_dsn) as db:
            folders = list(db.scalars(select(PortalChatFolder)))
            counts = dict(
                db.execute(
                    select(PortalChat.folder_id, func.count())
                    .where(PortalChat.archived.is_(False))
                    .group_by(PortalChat.folder_id)
                ).all()
            )
            by_parent: dict[Optional[str], list[PortalChatFolder]] = {}
            for f in folders:
                by_parent.setdefault(f.parent_id, []).append(f)

            def _tree(parent: Optional[str]) -> list[dict]:
                nodes = sorted(
                    by_parent.get(parent, []), key=lambda f: (f.position, f.name.lower())
                )
                return [
                    {**_folder_out(f), "chat_count": counts.get(f.id, 0), "children": _tree(f.id)}
                    for f in nodes
                ]

            return {"folders": _tree(None), "unfiled_count": counts.get(None, 0)}

    @router.post("/folders", status_code=201)
    def folders_create(body: FolderCreate, cfg: Any = Depends(get_cfg)) -> dict:
        name = body.name.strip()
        if not name:
            raise HTTPException(status_code=422, detail="name is required")
        with _get_session(cfg.db_dsn) as db:
            if body.parent_id and db.get(PortalChatFolder, body.parent_id) is None:
                raise HTTPException(status_code=404, detail="Parent folder not found")
            now = _now()
            folder = PortalChatFolder(
                id=_new_id(),
                parent_id=body.parent_id,
                name=name,
                position=body.position,
                created_at=now,
                updated_at=now,
            )
            db.add(folder)
            db.flush()
            return _folder_out(folder)

    @router.patch("/folders/{folder_id}")
    def folders_patch(folder_id: str, body: FolderPatch, cfg: Any = Depends(get_cfg)) -> dict:
        with _get_session(cfg.db_dsn) as db:
            folder = db.get(PortalChatFolder, folder_id)
            if folder is None:
                raise HTTPException(status_code=404, detail="Folder not found")
            if body.name is not None:
                if not body.name.strip():
                    raise HTTPException(status_code=422, detail="name is required")
                folder.name = body.name.strip()
            if body.to_root:
                folder.parent_id = None
            elif body.parent_id is not None:
                if body.parent_id == folder_id or body.parent_id in _folder_descendants(
                    db, folder_id
                ):
                    raise HTTPException(
                        status_code=422, detail="A folder cannot move inside itself"
                    )
                if db.get(PortalChatFolder, body.parent_id) is None:
                    raise HTTPException(status_code=404, detail="Parent folder not found")
                folder.parent_id = body.parent_id
            if body.position is not None:
                folder.position = body.position
            folder.updated_at = _now()
            db.flush()
            return _folder_out(folder)

    @router.delete("/folders/{folder_id}")
    def folders_delete(folder_id: str, cfg: Any = Depends(get_cfg)) -> dict:
        """Delete a folder; its chats and subfolders move up to its parent."""
        with _get_session(cfg.db_dsn) as db:
            folder = db.get(PortalChatFolder, folder_id)
            if folder is None:
                raise HTTPException(status_code=404, detail="Folder not found")
            parent = folder.parent_id
            db.execute(
                update(PortalChat).where(PortalChat.folder_id == folder_id).values(folder_id=parent)
            )
            db.execute(
                update(PortalChatFolder)
                .where(PortalChatFolder.parent_id == folder_id)
                .values(parent_id=parent)
            )
            db.delete(folder)
            return {"deleted": folder_id, "moved_to": parent}

    # ── tags & search ────────────────────────────────────────────────────────

    @router.get("/tags")
    def tags_tree(cfg: Any = Depends(get_cfg)) -> dict:
        with _get_session(cfg.db_dsn) as db:
            chat_tags = list(
                db.scalars(select(PortalChat.tags).where(PortalChat.archived.is_(False)))
            )
            clip_tags = list(db.scalars(select(PortalClip.tags)))
            ann_tags = list(db.scalars(select(PortalAnnotation.tags)))
            file_tags = list(file_tags_map(db, _tagged_file_keys(db)).values())
        return {
            "tree": build_tag_tree(chat_tags + clip_tags + ann_tags + file_tags),
            "chats": build_tag_tree(chat_tags),
            "files": build_tag_tree(file_tags),
        }

    @router.post("/tags/rename")
    async def tags_rename(body: TagRename, cfg: Any = Depends(get_cfg)) -> dict:
        src = "/".join(clean_tags([body.src]))
        dst = "/".join(clean_tags([body.dst]))
        if not src or not dst:
            raise HTTPException(status_code=422, detail="from and to are required")

        def _rename_rows() -> tuple[int, list[str]]:
            changed = 0
            with _get_session(cfg.db_dsn) as db:
                for model in (PortalChat, PortalClip, PortalAnnotation, PortalFileTags):
                    for row in db.scalars(select(model)):
                        new = rename_tag(row.tags, src, dst)
                        if new is not None:
                            row.tags = new
                            changed += 1
                notes = [
                    row.key
                    for row in db.scalars(select(PortalFileIndex))
                    if row.kind == "markdown" and tag_matches(row.tags, src)
                ]
            return changed, notes

        changed, notes = await asyncio.to_thread(_rename_rows)
        store = _store(cfg)
        touched: list[str] = []
        for key in notes:
            try:
                if await asyncio.to_thread(_rewrite_note_tags, store, key, src, dst):
                    touched.append(key)
            except VaultError:
                continue  # read-only note: its frontmatter stays as is
        await asyncio.to_thread(file_index.index_keys, cfg, notes, store)
        if touched:
            _publish(touched)
        return {"from": src, "to": dst, "changed": changed + len(touched)}

    @router.get("/search")
    def search(
        q: str = Query(min_length=1),
        limit: int = Query(default=30, ge=1, le=200),
        cfg: Any = Depends(get_cfg),
    ) -> dict:
        text, want_tags = split_query(q)
        like = f"%{text}%"
        needle = text.lower()

        def _snippet(body: str) -> str:
            if not needle:
                return body[:180].replace("\n", " ")
            idx = body.lower().find(needle)
            start = max(0, idx - 60) if idx >= 0 else 0
            return ("…" if start else "") + body[start : start + 180].replace("\n", " ")

        def _tagged(tags: Optional[list[str]]) -> bool:
            return all(tag_matches(tags, t) for t in want_tags)

        # With tag filters, fetch wider and filter in Python (tags are JSON).
        fetch = limit if not want_tags else max(limit * 10, 500)
        with _get_session(cfg.db_dsn) as db:
            chat_stmt = select(PortalChat)
            if text:
                chat_stmt = chat_stmt.where(PortalChat.title.ilike(like))
            chats = [c for c in db.scalars(chat_stmt.limit(fetch)) if _tagged(c.tags)][:limit]

            msgs: list[PortalMessage] = []
            if text:
                msg_stmt = (
                    select(PortalMessage)
                    .where(PortalMessage.content.ilike(like))
                    .order_by(PortalMessage.created_at.desc())
                    .limit(fetch)
                )
                msgs = list(db.scalars(msg_stmt))
            chat_rows = {
                c.id: c
                for c in db.scalars(
                    select(PortalChat).where(PortalChat.id.in_({m.chat_id for m in msgs}))
                )
            }
            msgs = [
                m for m in msgs if m.chat_id in chat_rows and _tagged(chat_rows[m.chat_id].tags)
            ][:limit]

            clip_stmt = select(PortalClip)
            if text:
                clip_stmt = clip_stmt.where(
                    or_(PortalClip.title.ilike(like), PortalClip.content_md.ilike(like))
                )
            clips = [c for c in db.scalars(clip_stmt.limit(fetch)) if _tagged(c.tags)][:limit]

            ann_stmt = select(PortalAnnotation)
            if text:
                # The note and the highlighted quote (anchor.exact) both match.
                ann_stmt = ann_stmt.where(
                    or_(
                        PortalAnnotation.body_md.ilike(like),
                        cast(PortalAnnotation.anchor, Text).ilike(like),
                    )
                )
            anns = [a for a in db.scalars(ann_stmt.limit(fetch)) if _tagged(a.tags)][:limit]
            ann_context = _annotation_context(db, anns)

            file_stmt = select(PortalFileIndex)
            if text:
                file_stmt = file_stmt.where(_file_match(db, text, like))
            file_stmt = file_stmt.order_by(PortalFileIndex.modified.desc()).limit(fetch)
            file_rows = [f for f in db.scalars(file_stmt) if _tagged(f.tags)][:limit]
            if want_tags and len(file_rows) < limit:
                # Files tagged only in the DB but not (yet) indexed.
                seen = {f.key for f in file_rows}
                for row in db.scalars(select(PortalFileTags)):
                    if row.key not in seen and _tagged(row.tags):
                        if not text or needle in row.key.lower():
                            file_rows.append(
                                PortalFileIndex(key=row.key, title="", kind="", tags=row.tags)
                            )

            sessions: list[dict[str, Any]] = []
            if text:
                try:
                    from pawn_server.core.sessions_api import list_sessions_page  # noqa: PLC0415

                    page = list_sessions_page(cfg, q=text, limit=limit)
                    sessions = [
                        {
                            "session_id": row["session_id"],
                            "title": row.get("title") or row["session_id"],
                            "snippet": _snippet(str(row.get("summary") or row.get("title") or "")),
                            "created_at": row.get("created_at"),
                        }
                        for row in page.get("data") or []
                    ][:limit]
                except Exception as exc:  # noqa: BLE001
                    logger.debug("session search: %s", exc)

            return {
                "query": text,
                "tags": want_tags,
                "chats": [_chat_out(c) for c in chats],
                "sessions": sessions,
                "messages": [
                    {
                        "id": m.id,
                        "chat_id": m.chat_id,
                        "chat_title": chat_rows[m.chat_id].title,
                        "role": m.role,
                        "seq": m.seq,
                        "snippet": _snippet(m.content),
                        "created_at": _iso(m.created_at),
                    }
                    for m in msgs
                ],
                "files": [
                    {
                        "key": f.key,
                        "title": f.title or f.key.rsplit("/", 1)[-1],
                        "kind": f.kind or file_index.file_kind(f.key),
                        "snippet": _snippet(f.content or "") if f.content else "",
                        "tags": list(f.tags or []),
                        "modified": _iso(f.modified),
                    }
                    for f in file_rows[:limit]
                ],
                "clips": [_clip_out(c) for c in clips],
                "annotations": [
                    {**_annotation_out(a), "context": ann_context.get(a.id, {})} for a in anns
                ],
            }

    # ── clips ────────────────────────────────────────────────────────────────

    @router.get("/clips")
    def clips_list(
        tag: Optional[str] = None, chat_id: Optional[str] = None, cfg: Any = Depends(get_cfg)
    ) -> dict:
        with _get_session(cfg.db_dsn) as db:
            stmt = select(PortalClip).order_by(PortalClip.created_at.desc())
            if chat_id:
                stmt = stmt.where(PortalClip.chat_id == chat_id)
            clips = list(db.scalars(stmt))
            if tag:
                clips = [c for c in clips if tag_matches(c.tags, tag)]
            return {"clips": [_clip_out(c) for c in clips]}

    @router.post("/clips", status_code=201)
    def clips_create(body: ClipCreate, cfg: Any = Depends(get_cfg)) -> dict:
        with _get_session(cfg.db_dsn) as db:
            chat = _get_chat(db, body.chat_id) if body.chat_id else None
            msgs: list[PortalMessage] = []
            if body.message_ids:
                found = {
                    m.id: m
                    for m in db.scalars(
                        select(PortalMessage).where(PortalMessage.id.in_(body.message_ids))
                    )
                }
                missing = [i for i in body.message_ids if i not in found]
                if missing:
                    raise HTTPException(status_code=404, detail=f"Unknown messages: {missing}")
                msgs = sorted(found.values(), key=lambda m: (m.chat_id, m.seq))
            if body.content_md is None and not msgs:
                raise HTTPException(status_code=422, detail="message_ids or content_md required")
            content = body.content_md if body.content_md is not None else _clip_markdown(chat, msgs)
            now = _now()
            clip = PortalClip(
                id=_new_id(),
                chat_id=body.chat_id or (msgs[0].chat_id if msgs else None),
                message_ids=[m.id for m in msgs] or list(body.message_ids),
                title=body.title.strip() or _title_from(content) or "Clip",
                tags=clean_tags(body.tags),
                content_md=content,
                created_at=now,
                updated_at=now,
            )
            db.add(clip)
            db.flush()
            return _clip_out(clip)

    @router.get("/clips/{clip_id}")
    def clips_get(clip_id: str, cfg: Any = Depends(get_cfg)) -> dict:
        with _get_session(cfg.db_dsn) as db:
            clip = db.get(PortalClip, clip_id)
            if clip is None:
                raise HTTPException(status_code=404, detail="Clip not found")
            return _clip_out(clip)

    @router.patch("/clips/{clip_id}")
    def clips_patch(clip_id: str, body: ClipPatch, cfg: Any = Depends(get_cfg)) -> dict:
        with _get_session(cfg.db_dsn) as db:
            clip = db.get(PortalClip, clip_id)
            if clip is None:
                raise HTTPException(status_code=404, detail="Clip not found")
            if body.title is not None:
                clip.title = body.title.strip()
            if body.tags is not None:
                clip.tags = clean_tags(body.tags)
            if body.content_md is not None:
                clip.content_md = body.content_md
            clip.updated_at = _now()
            db.flush()
            return _clip_out(clip)

    @router.delete("/clips/{clip_id}")
    def clips_delete(clip_id: str, cfg: Any = Depends(get_cfg)) -> dict:
        with _get_session(cfg.db_dsn) as db:
            clip = db.get(PortalClip, clip_id)
            if clip is None:
                raise HTTPException(status_code=404, detail="Clip not found")
            db.delete(clip)
        return {"deleted": clip_id}

    @router.post("/clips/{clip_id}/save")
    async def clips_save(clip_id: str, body: ClipSave, cfg: Any = Depends(get_cfg)) -> dict:
        with _get_session(cfg.db_dsn) as db:
            clip = db.get(PortalClip, clip_id)
            if clip is None:
                raise HTTPException(status_code=404, detail="Clip not found")
            title, content, tags = clip.title, clip.content_md, list(clip.tags or [])
        key = normalize_vault_key(
            body.path or f"{scratch_dir(cfg)}/Clips/{_safe_filename(title)}.md"
        )
        from pawn_core.vault import dump_frontmatter  # noqa: PLC0415

        meta: dict[str, Any] = {"pawn": "editable", "source": "portal-clip", "clip_id": clip_id}
        if tags:
            meta["tags"] = tags
        store = _store(cfg)
        try:
            stat = await asyncio.to_thread(
                store.write, key, dump_frontmatter(meta, f"# {title}\n\n{content}")
            )
        except VaultError as exc:
            raise _vault_http_error(exc) from exc
        with _get_session(cfg.db_dsn) as db:
            clip = db.get(PortalClip, clip_id)
            if clip is not None:
                clip.vault_key = stat.key
                clip.updated_at = _now()
        _publish([stat.key])
        return {"key": stat.key, "etag": stat.etag}

    # ── annotations ──────────────────────────────────────────────────────────

    @router.get("/annotations")
    def annotations_list(
        target_kind: Optional[str] = None,
        target_ref: Optional[list[str]] = Query(default=None),
        tag: Optional[str] = None,
        cfg: Any = Depends(get_cfg),
    ) -> dict:
        with _get_session(cfg.db_dsn) as db:
            stmt = select(PortalAnnotation).order_by(PortalAnnotation.created_at)
            if target_kind:
                stmt = stmt.where(PortalAnnotation.target_kind == target_kind)
            if target_ref:
                stmt = stmt.where(PortalAnnotation.target_ref.in_(target_ref))
            anns = list(db.scalars(stmt))
            if tag:
                anns = [a for a in anns if tag_matches(a.tags, tag)]
            return {"annotations": [_annotation_out(a) for a in anns]}

    @router.post("/annotations", status_code=201)
    def annotations_create(body: AnnotationCreate, cfg: Any = Depends(get_cfg)) -> dict:
        if body.target_kind not in TARGET_KINDS:
            raise HTTPException(
                status_code=422, detail=f"target_kind must be one of {TARGET_KINDS}"
            )
        if not body.target_ref.strip():
            raise HTTPException(status_code=422, detail="target_ref is required")
        with _get_session(cfg.db_dsn) as db:
            now = _now()
            ann = PortalAnnotation(
                id=_new_id(),
                target_kind=body.target_kind,
                target_ref=body.target_ref.strip(),
                anchor=body.anchor,
                body_md=body.body_md,
                tags=clean_tags(body.tags),
                color=body.color,
                side_chat_id=body.side_chat_id,
                created_at=now,
                updated_at=now,
            )
            db.add(ann)
            db.flush()
            return _annotation_out(ann)

    @router.patch("/annotations/{ann_id}")
    def annotations_patch(ann_id: str, body: AnnotationPatch, cfg: Any = Depends(get_cfg)) -> dict:
        with _get_session(cfg.db_dsn) as db:
            ann = db.get(PortalAnnotation, ann_id)
            if ann is None:
                raise HTTPException(status_code=404, detail="Annotation not found")
            for field in ("anchor", "body_md"):
                value = getattr(body, field)
                if value is not None:
                    setattr(ann, field, value)
            # An explicit null clears the colour or unlinks the side chat.
            for field in ("color", "side_chat_id"):
                if field in body.model_fields_set:
                    setattr(ann, field, getattr(body, field))
            if body.tags is not None:
                ann.tags = clean_tags(body.tags)
            ann.updated_at = _now()
            db.flush()
            return _annotation_out(ann)

    @router.delete("/annotations/{ann_id}")
    def annotations_delete(ann_id: str, cfg: Any = Depends(get_cfg)) -> dict:
        with _get_session(cfg.db_dsn) as db:
            ann = db.get(PortalAnnotation, ann_id)
            if ann is None:
                raise HTTPException(status_code=404, detail="Annotation not found")
            db.delete(ann)
        return {"deleted": ann_id}

    # ── files (S3 vault) ─────────────────────────────────────────────────────

    @router.get("/files")
    async def files_list(
        prefix: str = "", recursive: bool = False, cfg: Any = Depends(get_cfg)
    ) -> dict:
        store = _store(cfg)
        try:
            stats = await asyncio.to_thread(store.list_meta, prefix)
        except VaultError as exc:
            raise _vault_http_error(exc) from exc
        if recursive:
            return {
                "prefix": normalize_vault_key(prefix),
                "files": [s.key for s in stats if _visible_key(s.key)],
            }
        base = normalize_vault_key(prefix)
        direct = [s.key for s in stats if "/" not in s.key[len(base) + 1 if base else 0 :]]

        def _tags() -> dict[str, list[str]]:
            with _get_session(cfg.db_dsn) as db:
                return file_tags_map(db, direct)

        out = list_folder(stats, prefix, await asyncio.to_thread(_tags))
        out["scratch_dir"] = scratch_dir(cfg)
        out["agent_root"] = store.agent_root
        return out

    @router.get("/files/content")
    async def files_read(key: str, raw: bool = False, cfg: Any = Depends(get_cfg)) -> Any:
        k = normalize_vault_key(key)
        if not k or not _visible_key(k):
            raise HTTPException(status_code=404, detail="Not found")
        store = _store(cfg)
        try:
            stat = await asyncio.to_thread(store.stat, k)
            if stat is None:
                raise VaultNotFound(k)
            data = await asyncio.to_thread(store.read_bytes, k)
        except VaultError as exc:
            raise _vault_http_error(exc) from exc
        media = mimetypes.guess_type(k)[0] or "application/octet-stream"
        is_text = k.endswith(_TEXT_SUFFIXES) or media.startswith("text/")
        if raw or not is_text:
            return Response(content=data, media_type=media, headers={"ETag": f'"{stat.etag}"'})
        writable = True
        text = data.decode("utf-8", errors="replace")
        try:
            store.assert_writable(k, existing_body=text)
        except VaultWriteDenied:
            writable = False

        def _tags() -> list[str]:
            with _get_session(cfg.db_dsn) as db:
                return file_tags_map(db, [k]).get(k, [])

        tags = await asyncio.to_thread(_tags)
        if k.endswith(".md"):
            for t in file_index.frontmatter_tags(parse_frontmatter(text)[0]):
                if t not in tags:
                    tags.append(t)
        return {
            "key": k,
            "content": text,
            "etag": stat.etag,
            "size": stat.size,
            "last_modified": (
                _iso(stat.last_modified.replace(tzinfo=None)) if stat.last_modified else None
            ),
            "writable": writable,
            "tags": tags,
        }

    @router.put("/files/content")
    async def files_write(body: FileWrite, cfg: Any = Depends(get_cfg)) -> dict:
        k = normalize_vault_key(body.key)
        store = _store(cfg)
        try:
            if body.etag:
                stat = await asyncio.to_thread(store.write_if_unchanged, k, body.content, body.etag)
            else:
                if await asyncio.to_thread(store.exists, k):
                    raise VaultConflict(f"{k!r} already exists")
                stat = await asyncio.to_thread(store.write, k, body.content)
        except VaultError as exc:
            raise _vault_http_error(exc) from exc
        _publish([stat.key])
        file_index.schedule_index(cfg, [stat.key])
        return {"key": stat.key, "etag": stat.etag, "size": stat.size}

    @router.post("/files/move")
    async def files_move(body: FileMove, cfg: Any = Depends(get_cfg)) -> dict:
        store = _store(cfg)
        try:
            stat = await asyncio.to_thread(store.move, body.src, body.dst, overwrite=body.overwrite)
        except VaultError as exc:
            raise _vault_http_error(exc) from exc
        src = normalize_vault_key(body.src)
        await asyncio.to_thread(_move_file_tags, cfg, src, stat.key)
        _publish([src, stat.key])
        file_index.schedule_index(cfg, [src, stat.key])
        return {"key": stat.key, "etag": stat.etag}

    @router.delete("/files")
    async def files_delete(key: str, cfg: Any = Depends(get_cfg)) -> dict:
        k = normalize_vault_key(key)
        store = _store(cfg)
        try:
            await asyncio.to_thread(store.delete, k)
        except VaultError as exc:
            raise _vault_http_error(exc) from exc
        await asyncio.to_thread(_drop_file_rows, cfg, k)
        _publish([k])
        return {"deleted": k}

    @router.post("/files/upload", status_code=201)
    async def files_upload(
        file: UploadFile = File(...),
        folder: str = Form(default=""),
        cfg: Any = Depends(get_cfg),
    ) -> dict:
        name = _safe_filename(file.filename or "upload").replace(" ", "_") or "upload"
        if "." not in name and file.filename and "." in file.filename:
            name += "." + file.filename.rsplit(".", 1)[1]
        base = normalize_vault_key(folder) or scratch_dir(cfg)
        key = f"{base}/{name}"
        data = await file.read()
        store = _store(cfg)
        try:
            if key.endswith(".md"):
                stat = await asyncio.to_thread(store.write, key, data.decode("utf-8"))
            else:
                stat = await asyncio.to_thread(
                    store.write_bytes,
                    key,
                    data,
                    content_type=file.content_type or "application/octet-stream",
                )
        except (VaultError, UnicodeDecodeError) as exc:
            if isinstance(exc, UnicodeDecodeError):
                raise HTTPException(status_code=422, detail="Markdown must be UTF-8") from exc
            raise _vault_http_error(exc) from exc
        _publish([stat.key])
        file_index.schedule_index(cfg, [stat.key])
        return {"key": stat.key, "etag": stat.etag, "size": stat.size}

    @router.get("/files/tags")
    async def files_tags_get(key: str, cfg: Any = Depends(get_cfg)) -> dict:
        k = normalize_vault_key(key)
        store = _store(cfg)
        return await asyncio.to_thread(_file_tags_state, cfg, store, k)

    @router.put("/files/tags")
    async def files_tags_put(body: FileTagsPut, cfg: Any = Depends(get_cfg)) -> dict:
        k = normalize_vault_key(body.key)
        if not k or not _visible_key(k):
            raise HTTPException(status_code=404, detail="Not found")
        tags = clean_tags(body.tags)
        store = _store(cfg)
        try:
            out = await asyncio.to_thread(_set_file_tags, cfg, store, k, tags)
        except VaultError as exc:
            raise _vault_http_error(exc) from exc
        if out.get("source") == "frontmatter":
            _publish([k])
        return out

    @router.post("/files/reindex")
    async def files_reindex(cfg: Any = Depends(get_cfg)) -> dict:
        store = _store(cfg)
        try:
            return await asyncio.to_thread(file_index.reconcile, cfg, store)
        except VaultError as exc:
            raise _vault_http_error(exc) from exc

    # ── workspace ────────────────────────────────────────────────────────────

    @router.get("/workspace")
    def workspace_get(name: str = "default", cfg: Any = Depends(get_cfg)) -> dict:
        with _get_session(cfg.db_dsn) as db:
            ws = db.scalar(select(PortalWorkspace).where(PortalWorkspace.name == name))
            if ws is None:
                return {"name": name, "layout": None, "updated_at": None}
            return {"name": ws.name, "layout": ws.layout, "updated_at": _iso(ws.updated_at)}

    @router.put("/workspace")
    def workspace_put(body: WorkspacePut, cfg: Any = Depends(get_cfg)) -> dict:
        with _get_session(cfg.db_dsn) as db:
            ws = db.scalar(select(PortalWorkspace).where(PortalWorkspace.name == body.name))
            if ws is None:
                ws = PortalWorkspace(id=_new_id(), name=body.name)
                db.add(ws)
            ws.layout = body.layout
            ws.updated_at = _now()
            db.flush()
            return {"name": ws.name, "layout": ws.layout, "updated_at": _iso(ws.updated_at)}

    return router


def _publish(paths: list[str]) -> None:
    """Tell the files sidebar (and other clients) that the portal wrote these keys."""
    from pawn_server.core.vault_events import publish_vault_event  # noqa: PLC0415

    publish_vault_event(paths, source="portal")
