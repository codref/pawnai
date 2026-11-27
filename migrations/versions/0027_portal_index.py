"""Portal: file index, file tags, chat auto-naming flags.

Revision ID: 0027
Revises: 0026
Create Date: 2026-11-27

``portal_file_index`` holds text extracted from vault files (Markdown, PDF,
DOCX, image captions) for portal search; ``portal_file_tags`` holds tags for
files whose body can't carry frontmatter. ``portal_chats`` gains the flags
that let the background model name and tag a chat until the user edits it.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0027"
down_revision: Union[str, None] = "0026"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "portal_chats",
        sa.Column("title_auto", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.add_column(
        "portal_chats",
        sa.Column("tags_auto", sa.Boolean(), nullable=False, server_default=sa.true()),
    )
    op.add_column("portal_chats", sa.Column("autoname_seq", sa.Integer(), nullable=True))

    op.create_table(
        "portal_file_tags",
        sa.Column("key", sa.String(), primary_key=True),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )

    op.create_table(
        "portal_file_index",
        sa.Column("key", sa.String(), primary_key=True),
        sa.Column("etag", sa.String(), nullable=False, server_default=""),
        sa.Column("size", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("modified", sa.DateTime(), nullable=True),
        sa.Column("kind", sa.String(), nullable=False, server_default="other"),
        sa.Column("title", sa.String(), nullable=False, server_default=""),
        sa.Column("content", sa.Text(), nullable=False, server_default=""),
        sa.Column("tags", sa.JSON(), nullable=True),
        sa.Column("indexed_at", sa.DateTime(), nullable=True),
        sa.Column("error", sa.String(), nullable=True),
    )
    op.create_index("ix_portal_file_index_modified", "portal_file_index", ["modified"])
    op.execute(
        "CREATE INDEX ix_portal_file_index_fts ON portal_file_index "
        "USING gin (to_tsvector('simple', title || ' ' || content))"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_portal_file_index_fts")
    op.drop_table("portal_file_index")
    op.drop_table("portal_file_tags")
    op.drop_column("portal_chats", "autoname_seq")
    op.drop_column("portal_chats", "tags_auto")
    op.drop_column("portal_chats", "title_auto")
