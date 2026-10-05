"""Add siyuan_agent_requests table for @pawn watcher lifecycle.

Revision ID: 0012
Revises: 0011
Create Date: 2026-09-21
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: Union[str, None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "siyuan_agent_requests",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("trigger_block_id", sa.String(), nullable=False),
        sa.Column("parent_block_id", sa.String(), nullable=False),
        sa.Column("root_id", sa.String(), nullable=False),
        sa.Column("notebook_id", sa.String(), nullable=False),
        sa.Column("instruction_hash", sa.String(), nullable=False),
        sa.Column("source_updated", sa.String(), nullable=True),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("conversation_id", sa.String(), nullable=False),
        sa.Column("output_block_id", sa.String(), nullable=True),
        sa.Column("agent_run_id", sa.String(), nullable=True),
        sa.Column("matrix_notify_id", sa.String(), nullable=True),
        sa.Column("instruction_text", sa.Text(), nullable=True),
        sa.Column("error_code", sa.String(), nullable=True),
        sa.Column("indexed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "trigger_block_id",
            "instruction_hash",
            name="uq_siyuan_agent_request_trigger_hash",
        ),
    )
    op.create_index(
        "ix_siyuan_agent_requests_status",
        "siyuan_agent_requests",
        ["status"],
    )
    op.create_index(
        "ix_siyuan_agent_requests_conversation_id",
        "siyuan_agent_requests",
        ["conversation_id"],
    )
    op.create_table(
        "siyuan_watcher_state",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("poll_watermark", sa.String(), nullable=False, server_default=""),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )


def downgrade() -> None:
    op.drop_table("siyuan_watcher_state")
    op.drop_index("ix_siyuan_agent_requests_conversation_id", table_name="siyuan_agent_requests")
    op.drop_index("ix_siyuan_agent_requests_status", table_name="siyuan_agent_requests")
    op.drop_table("siyuan_agent_requests")
