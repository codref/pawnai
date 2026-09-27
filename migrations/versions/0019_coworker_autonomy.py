"""Run lineage, schedule output notes, and autonomy audit.

Revision ID: 0019
Revises: 0018
Create Date: 2026-09-27
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0019"
down_revision: Union[str, None] = "0018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("agent_runs", sa.Column("parent_run_id", sa.String(), nullable=True))
    op.add_column(
        "agent_runs",
        sa.Column("depth", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("agent_runs", sa.Column("event_id", sa.String(), nullable=True))
    op.create_index("ix_agent_runs_parent_run_id", "agent_runs", ["parent_run_id"])
    op.create_index("ix_agent_runs_event_id", "agent_runs", ["event_id"])
    op.add_column("agent_schedules", sa.Column("output_note", sa.String(), nullable=True))
    op.create_table(
        "coworker_decisions",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("event_id", sa.String(), nullable=True),
        sa.Column("event_kind", sa.String(), nullable=False),
        sa.Column("proposed_action", sa.Text(), nullable=True),
        sa.Column("policy_decision", sa.String(), nullable=False),
        sa.Column("outcome", sa.Text(), nullable=True),
        sa.Column("agent_run_id", sa.String(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_coworker_decisions_event_id", "coworker_decisions", ["event_id"])


def downgrade() -> None:
    op.drop_index("ix_coworker_decisions_event_id", table_name="coworker_decisions")
    op.drop_table("coworker_decisions")
    op.drop_column("agent_schedules", "output_note")
    op.drop_index("ix_agent_runs_event_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_parent_run_id", table_name="agent_runs")
    op.drop_column("agent_runs", "event_id")
    op.drop_column("agent_runs", "depth")
    op.drop_column("agent_runs", "parent_run_id")
