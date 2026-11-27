"""Portal: chats, folders, message mirror, clips, annotations, workspace.

Revision ID: 0026
Revises: 0025
Create Date: 2026-11-27

Backs the chat-first web portal (``/portal``, ``/v1/portal/*``). See
docs/PORTAL.md.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0026"
down_revision: Union[str, None] = "0025"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "portal_chat_folders",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "parent_id",
            sa.String(),
            sa.ForeignKey("portal_chat_folders.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_portal_chat_folders_parent_id", "portal_chat_folders", ["parent_id"])

    op.create_table(
        "portal_chats",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("conversation_key", sa.String(), nullable=False, unique=True),
        sa.Column("title", sa.String(), nullable=False, server_default=""),
        sa.Column(
            "folder_id",
            sa.String(),
            sa.ForeignKey("portal_chat_folders.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("position", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("pinned", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("archived", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("parent_chat_id", sa.String(), nullable=True),
        sa.Column("fork_from_message_id", sa.String(), nullable=True),
        sa.Column("origin", sa.JSON(), nullable=True),
        sa.Column("last_message_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_portal_chats_folder_id", "portal_chats", ["folder_id"])
    op.create_index("ix_portal_chats_archived", "portal_chats", ["archived"])
    op.create_index("ix_portal_chats_parent_chat_id", "portal_chats", ["parent_chat_id"])

    op.create_table(
        "portal_messages",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "chat_id",
            sa.String(),
            sa.ForeignKey("portal_chats.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("seq", sa.Integer(), nullable=False),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("context", sa.JSON(), nullable=True),
        sa.Column("run_id", sa.String(), nullable=True),
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_portal_messages_chat_id", "portal_messages", ["chat_id"])
    # Full-text search over message bodies (Postgres only; SQLite uses LIKE).
    op.execute(
        "CREATE INDEX ix_portal_messages_fts ON portal_messages "
        "USING gin (to_tsvector('simple', content))"
    )

    op.create_table(
        "portal_clips",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("chat_id", sa.String(), nullable=True),
        sa.Column("message_ids", sa.JSON(), nullable=True),
        sa.Column("title", sa.String(), nullable=False, server_default=""),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("content_md", sa.Text(), nullable=False, server_default=""),
        sa.Column("vault_key", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_portal_clips_chat_id", "portal_clips", ["chat_id"])

    op.create_table(
        "portal_annotations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("target_kind", sa.String(), nullable=False),
        sa.Column("target_ref", sa.String(), nullable=False),
        sa.Column("anchor", sa.JSON(), nullable=True),
        sa.Column("body_md", sa.Text(), nullable=False, server_default=""),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("color", sa.String(), nullable=True),
        sa.Column("side_chat_id", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_portal_annotations_target_kind", "portal_annotations", ["target_kind"])
    op.create_index("ix_portal_annotations_target_ref", "portal_annotations", ["target_ref"])

    op.create_table(
        "portal_workspaces",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False, server_default="default"),
        sa.Column("layout", sa.JSON(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("portal_workspaces")
    op.drop_table("portal_annotations")
    op.drop_table("portal_clips")
    op.execute("DROP INDEX IF EXISTS ix_portal_messages_fts")
    op.drop_table("portal_messages")
    op.drop_table("portal_chats")
    op.drop_table("portal_chat_folders")
