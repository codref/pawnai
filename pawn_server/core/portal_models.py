"""ORM models for the Pawn portal (chats, folders, clips, annotations, workspace).

The portal keeps its own transcript mirror (``portal_messages``) so messages
have stable ids for annotations, clips and forks, and the raw user text stays
separate from the context injected into the agent prompt. sallm remains the
model's memory; nothing here is read by the agent loop. See docs/PORTAL.md.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from sqlalchemy import JSON, Boolean, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from pawn_core.database import Base


class PortalChatFolder(Base):
    """A (nestable) folder that groups saved chats in the left sidebar."""

    __tablename__ = "portal_chat_folders"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    parent_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("portal_chat_folders.id", ondelete="SET NULL"), nullable=True, index=True
    )
    name: Mapped[str] = mapped_column(String, nullable=False)
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class PortalChat(Base):
    """One portal conversation; ``conversation_key`` is the sallm session id."""

    __tablename__ = "portal_chats"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    conversation_key: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    title: Mapped[str] = mapped_column(String, nullable=False, default="")
    folder_id: Mapped[Optional[str]] = mapped_column(
        String, ForeignKey("portal_chat_folders.id", ondelete="SET NULL"), nullable=True, index=True
    )
    position: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    tags: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    parent_chat_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    fork_from_message_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # Side chats: what the chat was spawned from ({kind, ref, quote, anchor}).
    origin: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    # Auto-naming: the background model may (re)write title/tags until the
    # user edits them; autoname_seq is the message seq it last ran at.
    title_auto: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    tags_auto: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    autoname_seq: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    last_message_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class PortalMessage(Base):
    """One user or assistant turn as the portal shows it."""

    __tablename__ = "portal_messages"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    chat_id: Mapped[str] = mapped_column(
        String, ForeignKey("portal_chats.id", ondelete="CASCADE"), nullable=False, index=True
    )
    seq: Mapped[int] = mapped_column(Integer, nullable=False)
    role: Mapped[str] = mapped_column(String, nullable=False)
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    # Notes, selection, image metadata, origin, written_files.
    context: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    model: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class PortalClip(Base):
    """A saved portion of a chat (a few messages, or a selection across them)."""

    __tablename__ = "portal_clips"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    chat_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    message_ids: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    title: Mapped[str] = mapped_column(String, nullable=False, default="")
    tags: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    content_md: Mapped[str] = mapped_column(Text, nullable=False, default="")
    vault_key: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class PortalAnnotation(Base):
    """Highlight + note on any target (message, file, item, session, clip, capture)."""

    __tablename__ = "portal_annotations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    target_kind: Mapped[str] = mapped_column(String, nullable=False, index=True)
    target_ref: Mapped[str] = mapped_column(String, nullable=False, index=True)
    # Text-quote selector {exact, prefix, suffix} plus optional {start, end} hint.
    anchor: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    body_md: Mapped[str] = mapped_column(Text, nullable=False, default="")
    tags: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    color: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    side_chat_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class PortalWorkspace(Base):
    """Serialized dockview layout (tiles, floating windows, pins)."""

    __tablename__ = "portal_workspaces"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, default="default")
    layout: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class PortalBoard(Base):
    """One brainmap board: what is on it (seed + pins - hidden) and its layout.

    ``name`` is the board id used in URLs (``items`` is built in; user boards
    are ``b-…``). Meaning (links, parents, subjects) stays in the emergence
    tables; a board only stores membership, note cards and positions.
    """

    __tablename__ = "portal_boards"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False, unique=True)
    title: Mapped[str] = mapped_column(String, nullable=False, default="")
    # items | lens | manual | question
    kind: Mapped[str] = mapped_column(String, nullable=False, default="manual")
    # Lens seed: {q, kinds, statuses, tags, people, subject_ids, sources}
    spec: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    pins: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    hidden: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    # Accepted note cards: [{id, title, body, sources: [node_id]}]
    notes: Mapped[Optional[list[dict[str, Any]]]] = mapped_column(JSON, nullable=True)
    folder_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    tags: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    pinned: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    archived: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    chat_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    # Question boards: the em_subjects row the night shift works on.
    subject_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    # {node_id: {x, y, pinned}}
    positions: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    # {x, y, zoom}
    viewport: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class PortalBoardProposal(Base):
    """A ghost card / string / group / pin proposed for a board, until decided."""

    __tablename__ = "portal_board_proposals"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    board: Mapped[str] = mapped_column(String, nullable=False, index=True)
    chat_id: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    run_id: Mapped[Optional[str]] = mapped_column(String, nullable=True, index=True)
    # chat | night
    origin: Mapped[str] = mapped_column(String, nullable=False, default="chat")
    # string | card | group | pin
    kind: Mapped[str] = mapped_column(String, nullable=False)
    payload: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    # proposed | accepted | rejected
    status: Mapped[str] = mapped_column(String, nullable=False, default="proposed")
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    decided_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class PortalFileTags(Base):
    """Tags for vault files whose body Pawn can't or shouldn't rewrite."""

    __tablename__ = "portal_file_tags"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    tags: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    updated_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)


class PortalFileIndex(Base):
    """Searchable text extracted from one vault file (see file_index.py)."""

    __tablename__ = "portal_file_index"

    key: Mapped[str] = mapped_column(String, primary_key=True)
    etag: Mapped[str] = mapped_column(String, nullable=False, default="")
    size: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    modified: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True, index=True)
    kind: Mapped[str] = mapped_column(String, nullable=False, default="other")
    title: Mapped[str] = mapped_column(String, nullable=False, default="")
    content: Mapped[str] = mapped_column(Text, nullable=False, default="")
    tags: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(String, nullable=True)


PORTAL_TABLES = [
    PortalChatFolder.__table__,
    PortalChat.__table__,
    PortalMessage.__table__,
    PortalClip.__table__,
    PortalAnnotation.__table__,
    PortalWorkspace.__table__,
    PortalBoard.__table__,
    PortalBoardProposal.__table__,
    PortalFileTags.__table__,
    PortalFileIndex.__table__,
]
