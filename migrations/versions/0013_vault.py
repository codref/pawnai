"""Vault notes/tasks; drop legacy SiYuan tables.

Revision ID: 0013
Revises: 0012
Create Date: 2026-09-25
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0013"
down_revision: Union[str, None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_SIYUAN_TABLES = (
    "siyuan_watcher_state",
    "siyuan_agent_requests",
    "siyuan_session_docs",
)


def _drop_table_if_exists(name: str) -> None:
    bind = op.get_bind()
    inspector = sa.inspect(bind)
    if name in inspector.get_table_names():
        op.drop_table(name)


def upgrade() -> None:
    for table in _SIYUAN_TABLES:
        _drop_table_if_exists(table)

    op.create_table(
        "vault_notes",
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False, server_default=""),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("session_id"),
    )
    op.create_table(
        "vault_tasks",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("key", sa.String(), nullable=False),
        sa.Column("status", sa.String(), nullable=False),
        sa.Column("conversation_id", sa.String(), nullable=False),
        sa.Column("instruction_hash", sa.String(), nullable=False),
        sa.Column("etag", sa.String(), nullable=True),
        sa.Column("via", sa.String(), nullable=False, server_default="vault"),
        sa.Column("agent_run_id", sa.String(), nullable=True),
        sa.Column("matrix_notify_id", sa.String(), nullable=True),
        sa.Column("instruction_text", sa.Text(), nullable=True),
        sa.Column("note_path", sa.String(), nullable=True),
        sa.Column("error_code", sa.String(), nullable=True),
        sa.Column("indexed_at", sa.DateTime(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key", name="uq_vault_tasks_key"),
    )
    op.create_index("ix_vault_tasks_status", "vault_tasks", ["status"])
    op.create_index("ix_vault_tasks_conversation_id", "vault_tasks", ["conversation_id"])


def downgrade() -> None:
    op.drop_index("ix_vault_tasks_conversation_id", table_name="vault_tasks")
    op.drop_index("ix_vault_tasks_status", table_name="vault_tasks")
    op.drop_table("vault_tasks")
    op.drop_table("vault_notes")

    op.create_table(
        "siyuan_session_docs",
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("doc_id", sa.String(), nullable=False),
        sa.Column("path", sa.String(), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False, server_default=""),
        sa.Column("daily_linked", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("session_id"),
    )
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
