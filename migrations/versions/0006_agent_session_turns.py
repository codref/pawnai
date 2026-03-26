"""Add agent_session_turns table for persistent conversation history.

Revision ID: 0006
Revises: 0005
Create Date: 2026-03-24

Each row stores the new_messages() from one agent turn, keyed by an
opaque source_id (queue message_id or uuid4).  INSERT ... ON CONFLICT
DO NOTHING on source_id makes all writes idempotent.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0006"
down_revision: Union[str, None] = "0005"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_session_turns",
        sa.Column("source_id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("messages", JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("source_id"),
        if_not_exists=True,
    )
    op.create_index(
        "ix_agent_session_turns_session_id",
        "agent_session_turns",
        ["session_id", "created_at"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index(
        "ix_agent_session_turns_session_id",
        table_name="agent_session_turns",
    )
    op.drop_table("agent_session_turns")
