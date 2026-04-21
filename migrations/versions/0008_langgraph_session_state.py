"""Add langgraph_session_state table for LangGraph HTTP session persistence.

Revision ID: 0008
Revises: 0007
Create Date: 2026-04-21

Stores the persistent portion of a LangGraph chat session (durable_facts,
artifacts, recent_messages) so that HTTP sessions survive server restarts
and are shared across worker processes.

Transient per-turn fields (incoming_prompt, route_kind, action_plan, …)
are NOT persisted — they are always reset at the start of each turn.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0008"
down_revision: Union[str, None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "langgraph_session_state",
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column(
            "durable_facts",
            JSONB(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "artifacts",
            JSONB(),
            nullable=False,
            server_default="{}",
        ),
        sa.Column(
            "recent_messages",
            JSONB(),
            nullable=False,
            server_default="[]",
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("session_id"),
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_table("langgraph_session_state")
