"""Add agent_session_vars table for persistent session variable storage.

Revision ID: 0007
Revises: 0006
Create Date: 2026-04-12

Each row stores one session variable (key/value pair) for a named session.
The composite primary key (session_id, key) enforces uniqueness and enables
upsert via INSERT ... ON CONFLICT DO UPDATE.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0007"
down_revision: Union[str, None] = "0006"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_session_vars",
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("value", JSONB(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("session_id", "key"),
        if_not_exists=True,
    )
    op.create_index(
        "ix_agent_session_vars_session_id",
        "agent_session_vars",
        ["session_id"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_session_vars_session_id",
        table_name="agent_session_vars",
    )
    op.drop_table("agent_session_vars")
