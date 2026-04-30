"""Add durable agent schedules.

Revision ID: 0010
Revises: 0009
Create Date: 2026-04-30
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0010"
down_revision: Union[str, None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("agent_runs", sa.Column("source", sa.String(), nullable=True))
    op.add_column("agent_runs", sa.Column("schedule_id", sa.String(), nullable=True))
    op.add_column("agent_runs", sa.Column("scheduled_fire_id", sa.String(), nullable=True))
    op.execute("UPDATE agent_runs SET source = 'queue' WHERE source IS NULL")
    op.alter_column("agent_runs", "source", nullable=False)
    op.create_index("ix_agent_runs_source", "agent_runs", ["source"], if_not_exists=True)
    op.create_index("ix_agent_runs_schedule_id", "agent_runs", ["schedule_id"], if_not_exists=True)
    op.create_index(
        "ix_agent_runs_scheduled_fire_id",
        "agent_runs",
        ["scheduled_fire_id"],
        if_not_exists=True,
    )

    op.create_table(
        "agent_schedules",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("schedule_kind", sa.String(), nullable=False),
        sa.Column("timezone", sa.String(), nullable=False),
        sa.Column("run_at", sa.DateTime(), nullable=True),
        sa.Column("interval_seconds", sa.Integer(), nullable=True),
        sa.Column("cron_expression", sa.String(), nullable=True),
        sa.Column("next_run_at", sa.DateTime(), nullable=True),
        sa.Column("last_run_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.Column("created_by", sa.String(), nullable=False),
        sa.Column("created_from_proposal_id", sa.String(), nullable=True),
        sa.Column("revision", sa.Integer(), nullable=False),
        sa.Column("metadata", JSONB(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index("ix_agent_schedules_status", "agent_schedules", ["status"], if_not_exists=True)
    op.create_index(
        "ix_agent_schedules_session_id",
        "agent_schedules",
        ["session_id"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_agent_schedules_next_run_at",
        "agent_schedules",
        ["next_run_at"],
        if_not_exists=True,
    )

    op.create_table(
        "agent_schedule_proposals",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("action", sa.String(), nullable=False),
        sa.Column("schedule_id", sa.String(), nullable=True),
        sa.Column("proposed_payload", JSONB(), nullable=False),
        sa.Column("rationale", sa.Text(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("proposed_by_run_id", sa.String(), nullable=True),
        sa.Column("proposed_by_session_id", sa.String(), nullable=True),
        sa.Column("reviewed_by", sa.String(), nullable=True),
        sa.Column("reviewed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["schedule_id"], ["agent_schedules.id"]),
        sa.ForeignKeyConstraint(["proposed_by_run_id"], ["agent_runs.id"]),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index(
        "ix_agent_schedule_proposals_action",
        "agent_schedule_proposals",
        ["action"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_agent_schedule_proposals_status",
        "agent_schedule_proposals",
        ["status"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_agent_schedule_proposals_schedule_id",
        "agent_schedule_proposals",
        ["schedule_id"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_agent_schedule_proposals_proposed_by_session_id",
        "agent_schedule_proposals",
        ["proposed_by_session_id"],
        if_not_exists=True,
    )

    op.create_table(
        "agent_schedule_fires",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("schedule_id", sa.String(), nullable=False),
        sa.Column("scheduled_for", sa.DateTime(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("agent_run_id", sa.String(), nullable=True),
        sa.Column("attempt", sa.Integer(), nullable=False),
        sa.Column("claimed_at", sa.DateTime(), nullable=True),
        sa.Column("started_at", sa.DateTime(), nullable=True),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.ForeignKeyConstraint(["agent_run_id"], ["agent_runs.id"]),
        sa.ForeignKeyConstraint(["schedule_id"], ["agent_schedules.id"]),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("schedule_id", "scheduled_for", name="uq_agent_schedule_fire_once"),
        if_not_exists=True,
    )
    op.create_index(
        "ix_agent_schedule_fires_schedule_id",
        "agent_schedule_fires",
        ["schedule_id"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_agent_schedule_fires_scheduled_for",
        "agent_schedule_fires",
        ["scheduled_for"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_agent_schedule_fires_status",
        "agent_schedule_fires",
        ["status"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_agent_schedule_fires_agent_run_id",
        "agent_schedule_fires",
        ["agent_run_id"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_agent_schedule_fires_agent_run_id", table_name="agent_schedule_fires")
    op.drop_index("ix_agent_schedule_fires_status", table_name="agent_schedule_fires")
    op.drop_index("ix_agent_schedule_fires_scheduled_for", table_name="agent_schedule_fires")
    op.drop_index("ix_agent_schedule_fires_schedule_id", table_name="agent_schedule_fires")
    op.drop_table("agent_schedule_fires")

    op.drop_index(
        "ix_agent_schedule_proposals_proposed_by_session_id",
        table_name="agent_schedule_proposals",
    )
    op.drop_index("ix_agent_schedule_proposals_schedule_id", table_name="agent_schedule_proposals")
    op.drop_index("ix_agent_schedule_proposals_status", table_name="agent_schedule_proposals")
    op.drop_index("ix_agent_schedule_proposals_action", table_name="agent_schedule_proposals")
    op.drop_table("agent_schedule_proposals")

    op.drop_index("ix_agent_schedules_next_run_at", table_name="agent_schedules")
    op.drop_index("ix_agent_schedules_session_id", table_name="agent_schedules")
    op.drop_index("ix_agent_schedules_status", table_name="agent_schedules")
    op.drop_table("agent_schedules")

    op.drop_index("ix_agent_runs_scheduled_fire_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_schedule_id", table_name="agent_runs")
    op.drop_index("ix_agent_runs_source", table_name="agent_runs")
    op.drop_column("agent_runs", "scheduled_fire_id")
    op.drop_column("agent_runs", "schedule_id")
    op.drop_column("agent_runs", "source")
