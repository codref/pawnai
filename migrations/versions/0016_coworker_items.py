"""Coworker inbox items and suppressions.

Revision ID: 0016
Revises: 0015
Create Date: 2026-09-27
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0016"
down_revision: Union[str, None] = "0015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "coworker_items",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("short_id", sa.String(), nullable=False),
        sa.Column("source_kind", sa.String(), nullable=False),
        sa.Column("source_ref", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("owner", sa.String(), nullable=True),
        sa.Column("due", sa.String(), nullable=True),
        sa.Column("quote", sa.Text(), nullable=True),
        sa.Column("thread", sa.String(), nullable=True),
        sa.Column("interrupt", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("fingerprint", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="new"),
        sa.Column("snooze_until", sa.DateTime(), nullable=True),
        sa.Column("note_key", sa.String(), nullable=True),
        sa.Column("payload", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_coworker_items_short_id", "coworker_items", ["short_id"], unique=True)
    op.create_index("ix_coworker_items_source", "coworker_items", ["source_kind", "source_ref"])
    op.create_index("ix_coworker_items_status", "coworker_items", ["status"])
    op.create_index("ix_coworker_items_fingerprint", "coworker_items", ["fingerprint"])
    op.create_index("ix_coworker_items_thread", "coworker_items", ["thread"])
    op.create_table(
        "coworker_suppressions",
        sa.Column("fingerprint", sa.String(), primary_key=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("coworker_suppressions")
    op.drop_index("ix_coworker_items_thread", table_name="coworker_items")
    op.drop_index("ix_coworker_items_fingerprint", table_name="coworker_items")
    op.drop_index("ix_coworker_items_status", table_name="coworker_items")
    op.drop_index("ix_coworker_items_source", table_name="coworker_items")
    op.drop_index("ix_coworker_items_short_id", table_name="coworker_items")
    op.drop_table("coworker_items")
