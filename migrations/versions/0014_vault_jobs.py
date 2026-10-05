"""Vault tasks become generic jobs (kind, payload, result_text).

Revision ID: 0014
Revises: 0013
Create Date: 2026-09-26
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0014"
down_revision: Union[str, None] = "0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "vault_tasks",
        sa.Column("kind", sa.String(), nullable=False, server_default="ask"),
    )
    op.add_column("vault_tasks", sa.Column("payload", sa.JSON(), nullable=True))
    op.add_column("vault_tasks", sa.Column("result_text", sa.Text(), nullable=True))
    op.create_index("ix_vault_tasks_created_at", "vault_tasks", ["created_at"])


def downgrade() -> None:
    op.drop_index("ix_vault_tasks_created_at", table_name="vault_tasks")
    op.drop_column("vault_tasks", "result_text")
    op.drop_column("vault_tasks", "payload")
    op.drop_column("vault_tasks", "kind")
