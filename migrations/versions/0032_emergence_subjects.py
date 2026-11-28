"""Emergence subjects: threads, questions, ledger, hypotheses, provocations.

Revision ID: 0032
Revises: 0031
Create Date: 2026-11-28

See docs/plans/EMERGENCE.md §3 and §5-6. ``em_subjects`` holds every
long-lived subject (thread, question, decision, commitment, prediction,
hypothesis) with a rewritten ``state_md`` and an embedding centroid.
``em_links`` attaches observations to subjects with a role.
``em_subject_events`` is the append-only history ("what did I believe in
March"). ``em_provocations`` are the 1-3 questions surfaced per day.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0032"
down_revision: Union[str, None] = "0031"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "em_subjects",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False),
        sa.Column("slug", sa.String(), nullable=False),
        sa.Column("origin", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("state_md", sa.Text(), nullable=True),
        sa.Column("fields", sa.JSON(), nullable=True),
        sa.Column("momentum", sa.Float(), nullable=True),
        sa.Column("source_ref", sa.String(), nullable=True),
        sa.Column("parent_id", sa.String(), nullable=True),
        sa.Column("last_evidence_at", sa.DateTime(), nullable=True),
        sa.Column("state_at", sa.DateTime(), nullable=True),
        sa.Column("checked_at", sa.DateTime(), nullable=True),
        sa.Column("review_at", sa.DateTime(), nullable=True),
        sa.Column("note_key", sa.String(), nullable=True),
        sa.Column("centroid", Vector(1024), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_em_subjects_kind", "em_subjects", ["kind"])
    op.create_index("ix_em_subjects_status", "em_subjects", ["status"])
    op.create_index("ix_em_subjects_slug", "em_subjects", ["slug"])

    op.create_table(
        "em_links",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("observation_id", sa.String(), nullable=False),
        sa.Column(
            "subject_id",
            sa.String(),
            sa.ForeignKey("em_subjects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("role", sa.String(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("linked_by", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("observation_id", "subject_id", name="uq_em_links_obs_subject"),
    )
    op.create_index("ix_em_links_subject_id", "em_links", ["subject_id"])
    op.create_index("ix_em_links_observation_id", "em_links", ["observation_id"])

    op.create_table(
        "em_subject_events",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column(
            "subject_id",
            sa.String(),
            sa.ForeignKey("em_subjects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("at", sa.DateTime(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("diff", sa.JSON(), nullable=True),
        sa.Column("cause_observation_ids", sa.JSON(), nullable=True),
    )
    op.create_index("ix_em_subject_events_subject_id", "em_subject_events", ["subject_id"])
    op.create_index("ix_em_subject_events_at", "em_subject_events", ["at"])

    op.create_table(
        "em_provocations",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("short_id", sa.String(), nullable=False, unique=True),
        sa.Column("day", sa.String(), nullable=False),
        sa.Column("subject_id", sa.String(), nullable=True),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("features", sa.JSON(), nullable=True),
        sa.Column("fingerprint", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False, server_default="candidate"),
        sa.Column("shown_at", sa.DateTime(), nullable=True),
        sa.Column("verdict", sa.String(), nullable=True),
        sa.Column("verdict_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_em_provocations_day", "em_provocations", ["day"])
    op.create_index("ix_em_provocations_status", "em_provocations", ["status"])
    op.create_index("ix_em_provocations_fingerprint", "em_provocations", ["fingerprint"])


def downgrade() -> None:
    op.drop_table("em_provocations")
    op.drop_table("em_subject_events")
    op.drop_table("em_links")
    op.drop_table("em_subjects")
