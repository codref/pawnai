"""Portal boards v2: user boards (lens/manual/question), board chats, proposals.

Revision ID: 0034
Revises: 0033
Create Date: 2026-11-28

``portal_boards`` gains a title, a kind, a seed spec, membership (pins,
hidden, note cards), sidebar fields and the board chat. Ghost cards and
strings the board chat (or the night shift) proposes live in
``portal_board_proposals`` until the user accepts or burns them.
See docs/PORTAL.md "Boards".
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0034"
down_revision: Union[str, None] = "0033"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    with op.batch_alter_table("portal_boards") as t:
        t.add_column(sa.Column("title", sa.String(), nullable=False, server_default=""))
        t.add_column(sa.Column("kind", sa.String(), nullable=False, server_default="manual"))
        t.add_column(sa.Column("spec", sa.JSON(), nullable=True))
        t.add_column(sa.Column("pins", sa.JSON(), nullable=True))
        t.add_column(sa.Column("hidden", sa.JSON(), nullable=True))
        t.add_column(sa.Column("notes", sa.JSON(), nullable=True))
        t.add_column(sa.Column("folder_id", sa.String(), nullable=True))
        t.add_column(sa.Column("tags", sa.JSON(), nullable=True))
        t.add_column(sa.Column("pinned", sa.Boolean(), nullable=False, server_default=sa.false()))
        t.add_column(sa.Column("archived", sa.Boolean(), nullable=False, server_default=sa.false()))
        t.add_column(sa.Column("chat_id", sa.String(), nullable=True))
        t.add_column(sa.Column("subject_id", sa.String(), nullable=True))
        t.add_column(sa.Column("seen_at", sa.DateTime(), nullable=True))
        t.add_column(sa.Column("created_at", sa.DateTime(), nullable=True))
        t.create_index("ix_portal_boards_folder_id", ["folder_id"])
        t.create_index("ix_portal_boards_subject_id", ["subject_id"])
    op.execute("UPDATE portal_boards SET kind = 'items', title = 'Items' WHERE name = 'items'")
    op.create_table(
        "portal_board_proposals",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("board", sa.String(), nullable=False, index=True),
        sa.Column("chat_id", sa.String(), nullable=True),
        sa.Column("run_id", sa.String(), nullable=True, index=True),
        sa.Column("origin", sa.String(), nullable=False, server_default="chat"),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="proposed"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("decided_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("portal_board_proposals")
    with op.batch_alter_table("portal_boards") as t:
        t.drop_index("ix_portal_boards_subject_id")
        t.drop_index("ix_portal_boards_folder_id")
        for col in (
            "created_at",
            "seen_at",
            "subject_id",
            "chat_id",
            "archived",
            "pinned",
            "tags",
            "folder_id",
            "notes",
            "hidden",
            "pins",
            "spec",
            "kind",
            "title",
        ):
            t.drop_column(col)
