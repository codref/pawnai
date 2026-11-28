"""Emergence labels: the user's verdicts, as self-contained training rows.

Revision ID: 0031
Revises: 0030
Create Date: 2026-11-28

See docs/plans/EMERGENCE.md §7. Each row snapshots what was judged (kind,
text, quote, context) plus its embedding, so labels survive re-runs that
supersede observations and can come from review notes whose items were
never stored. ``(target_kind, target_id)`` is unique: a later verdict on the
same target replaces the earlier one.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0031"
down_revision: Union[str, None] = "0030"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "em_labels",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("target_kind", sa.String(), nullable=False),
        sa.Column("target_id", sa.String(), nullable=False),
        sa.Column("verdict", sa.String(), nullable=False),
        sa.Column("surface", sa.String(), nullable=False),
        sa.Column("type", sa.String(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("quote", sa.Text(), nullable=True),
        sa.Column("context", sa.Text(), nullable=True),
        sa.Column("source_ref", sa.String(), nullable=True),
        sa.Column("start_sec", sa.Float(), nullable=True),
        sa.Column("observation_id", sa.String(), nullable=True),
        sa.Column("item_id", sa.String(), nullable=True),
        sa.Column("extractor", sa.String(), nullable=True),
        sa.Column("features", sa.JSON(), nullable=True),
        sa.Column("before", sa.JSON(), nullable=True),
        sa.Column("after", sa.JSON(), nullable=True),
        sa.Column("embedding", Vector(1024), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("target_kind", "target_id", name="uq_em_labels_target"),
    )
    op.create_index("ix_em_labels_type", "em_labels", ["type"])
    op.create_index("ix_em_labels_verdict", "em_labels", ["verdict"])
    op.create_index("ix_em_labels_source_ref", "em_labels", ["source_ref"])


def downgrade() -> None:
    op.drop_table("em_labels")
