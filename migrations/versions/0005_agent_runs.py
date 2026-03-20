"""Add agent_runs table for queue-initiated agent execution history.

Revision ID: 0005
Revises: 0004
Create Date: 2026-03-21
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0005"
down_revision: Union[str, None] = "0004"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "agent_runs",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("message_id", sa.String(), nullable=True),
        sa.Column("command", sa.String(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=True),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("response", sa.Text(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index(
        "ix_agent_runs_session_id",
        "agent_runs",
        ["session_id"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_agent_runs_status",
        "agent_runs",
        ["status"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_agent_runs_created_at",
        "agent_runs",
        ["created_at"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_agent_runs_created_at", table_name="agent_runs")
    op.drop_index("ix_agent_runs_status", table_name="agent_runs")
    op.drop_index("ix_agent_runs_session_id", table_name="agent_runs")
    op.drop_table("agent_runs")
