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
    PortalFileTags.__table__,
    PortalFileIndex.__table__,
]
