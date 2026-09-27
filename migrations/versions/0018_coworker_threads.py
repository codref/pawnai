"""Thread movement tracking and per-item movement flag.

Revision ID: 0018
Revises: 0017
Create Date: 2026-09-27
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0018"
down_revision: Union[str, None] = "0017"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "coworker_items",
        sa.Column("movement", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    op.create_table(
        "coworker_threads",
        sa.Column("slug", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="active"),
        sa.Column("last_movement_at", sa.DateTime(), nullable=True),
        sa.Column("last_mention_at", sa.DateTime(), nullable=True),
        sa.Column("open_items", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_coworker_threads_status", "coworker_threads", ["status"])


def downgrade() -> None:
    op.drop_index("ix_coworker_threads_status", table_name="coworker_threads")
    op.drop_table("coworker_threads")
    op.drop_column("coworker_items", "movement")
