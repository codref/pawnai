"""Emergence layer: sources, windows, observations (Phase 1).

Revision ID: 0029
Revises: 0028
Create Date: 2026-11-28

See docs/plans/EMERGENCE.md. ``em_sources`` has one row per processed source
(``kind``, ``ref``). ``em_windows`` holds the latest run's slices with triage
scores. ``em_observations`` keeps every run's observations; a re-run marks
the previous run's rows ``superseded``. Embedding columns are nullable and
filled by later phases.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0029"
down_revision: Union[str, None] = "0028"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "em_sources",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("ref", sa.String(), nullable=False),
        sa.Column("occurred_at", sa.DateTime(), nullable=True),
        sa.Column("content_hash", sa.String(), nullable=True),
        sa.Column("pipeline_version", sa.String(), nullable=True),
        sa.Column("run_id", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="done"),
        sa.Column("stats", sa.JSON(), nullable=True),
        sa.Column("processed_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("kind", "ref", name="uq_em_sources_kind_ref"),
    )
    op.create_index("ix_em_sources_ref", "em_sources", ["ref"])

    op.create_table(
        "em_windows",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "source_id",
            sa.String(),
            sa.ForeignKey("em_sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("ordinal", sa.Integer(), nullable=False),
        sa.Column("char_start", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("char_end", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("start_sec", sa.Float(), nullable=True),
        sa.Column("end_sec", sa.Float(), nullable=True),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("triage", sa.JSON(), nullable=True),
        sa.Column("triage_model", sa.String(), nullable=True),
        sa.Column("embedding", Vector(1024), nullable=True),
    )
    op.create_index("ix_em_windows_source_id", "em_windows", ["source_id"])
    op.create_index("ix_em_windows_content_hash", "em_windows", ["content_hash"])

    op.create_table(
        "em_observations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "source_id",
            sa.String(),
            sa.ForeignKey("em_sources.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "window_id",
            sa.String(),
            sa.ForeignKey("em_windows.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("run_id", sa.String(), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("context", sa.Text(), nullable=True),
        sa.Column("speaker_name", sa.String(), nullable=True),
        sa.Column("speaker_id", sa.String(), nullable=True),
        sa.Column("quote", sa.Text(), nullable=True),
        sa.Column("quote_score", sa.Float(), nullable=True),
        sa.Column("start_sec", sa.Float(), nullable=True),
        sa.Column("fields", sa.JSON(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("extractor", sa.String(), nullable=True),
        sa.Column("model", sa.String(), nullable=True),
        sa.Column("embedding", Vector(1024), nullable=True),
        sa.Column("status", sa.String(), nullable=False, server_default="live"),
        sa.Column("reject_reason", sa.String(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_em_observations_source_id", "em_observations", ["source_id"])
    op.create_index("ix_em_observations_run_id", "em_observations", ["run_id"])
    op.create_index("ix_em_observations_type", "em_observations", ["type"])
    op.create_index("ix_em_observations_status", "em_observations", ["status"])


def downgrade() -> None:
    op.drop_table("em_observations")
    op.drop_table("em_windows")
    op.drop_index("ix_em_sources_ref", table_name="em_sources")
    op.drop_table("em_sources")
