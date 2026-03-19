"""Add graph_triples table for knowledge-graph extraction results.

Revision ID: 0004
Revises: 0003
Create Date: 2026-03-19
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0004"
down_revision: Union[str, None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "graph_triples",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("subject", sa.Text(), nullable=False),
        sa.Column("relation", sa.Text(), nullable=False),
        sa.Column("object", sa.Text(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("extracted_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index(
        "ix_graph_triples_session_id",
        "graph_triples",
        ["session_id"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_graph_triples_session_subject",
        "graph_triples",
        ["session_id", "subject"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_graph_triples_session_subject", table_name="graph_triples")
    op.drop_index("ix_graph_triples_session_id", table_name="graph_triples")
    op.drop_table("graph_triples")
