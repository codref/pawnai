"""Add siyuan_session_docs mapping table.

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-21
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: Union[str, None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "siyuan_session_docs",
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column(
            "content_hash",
            sa.String(),
            nullable=False,
            server_default="",
        ),
        sa.Column(
            "daily_linked",
            sa.Boolean(),
            nullable=False,
            server_default=sa.false(),
        ),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("session_id"),
    )


def downgrade() -> None:
    op.drop_table("siyuan_session_docs")
