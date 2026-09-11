"""Add graph run events and topology snapshots.

Revision ID: 0011
Revises: 0010
Create Date: 2026-05-01
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0011"
down_revision: Union[str, None] = "0010"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "graph_topologies",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("graph_name", sa.String(), nullable=False),
        sa.Column("graph_version", sa.String(), nullable=False),
        sa.Column("topology", JSONB(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "graph_name",
            "graph_version",
            name="uq_graph_topologies_name_version",
        ),
        if_not_exists=True,
    )
    op.create_index(
        "ix_graph_topologies_graph_name",
        "graph_topologies",
        ["graph_name"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_graph_topologies_name_version",
        "graph_topologies",
        ["graph_name", "graph_version"],
        if_not_exists=True,
    )

    op.create_table(
        "graph_run_events",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("sequence", sa.Integer(), nullable=False),
        sa.Column("thread_id", sa.String(), nullable=True),
        sa.Column("trace_id", sa.String(), nullable=True),
        sa.Column("graph_name", sa.String(), nullable=False),
        sa.Column("graph_version", sa.String(), nullable=False),
        sa.Column("event_type", sa.String(), nullable=False),
        sa.Column("node_name", sa.String(), nullable=True),
        sa.Column("from_node", sa.String(), nullable=True),
        sa.Column("to_node", sa.String(), nullable=True),
        sa.Column("router_choice", sa.String(), nullable=True),
        sa.Column("timestamp", sa.DateTime(), nullable=False),
        sa.Column("duration_ms", sa.Integer(), nullable=True),
        sa.Column("status", sa.String(), nullable=True),
        sa.Column("payload", JSONB(), nullable=True),
        sa.ForeignKeyConstraint(["run_id"], ["agent_runs.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("run_id", "sequence", name="uq_graph_run_events_run_sequence"),
        if_not_exists=True,
    )
    op.create_index(
        "ix_graph_run_events_run_id",
        "graph_run_events",
        ["run_id"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_graph_run_events_run_id_sequence",
        "graph_run_events",
        ["run_id", "sequence"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_graph_run_events_run_id_timestamp",
        "graph_run_events",
        ["run_id", "timestamp"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_graph_run_events_event_type",
        "graph_run_events",
        ["event_type"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_graph_run_events_node_name",
        "graph_run_events",
        ["node_name"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_graph_run_events_thread_id",
        "graph_run_events",
        ["thread_id"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_graph_run_events_thread_id", table_name="graph_run_events")
    op.drop_index("ix_graph_run_events_node_name", table_name="graph_run_events")
    op.drop_index("ix_graph_run_events_event_type", table_name="graph_run_events")
    op.drop_index("ix_graph_run_events_run_id_timestamp", table_name="graph_run_events")
    op.drop_index("ix_graph_run_events_run_id_sequence", table_name="graph_run_events")
    op.drop_index("ix_graph_run_events_run_id", table_name="graph_run_events")
    op.drop_table("graph_run_events")
    op.drop_index("ix_graph_topologies_name_version", table_name="graph_topologies")
    op.drop_index("ix_graph_topologies_graph_name", table_name="graph_topologies")
    op.drop_table("graph_topologies")
