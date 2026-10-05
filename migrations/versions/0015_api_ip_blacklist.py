"""IP blacklist for API brute-force / scan protection.

Revision ID: 0015
Revises: 0014
Create Date: 2026-09-27
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0015"
down_revision: Union[str, None] = "0014"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "api_ip_blacklist",
        sa.Column("ip", sa.String(), primary_key=True, nullable=False),
        sa.Column("reason", sa.String(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.Column("expires_at", sa.DateTime(), nullable=True),
        sa.Column("hit_count", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_index("ix_api_ip_blacklist_expires_at", "api_ip_blacklist", ["expires_at"])


def downgrade() -> None:
    op.drop_index("ix_api_ip_blacklist_expires_at", table_name="api_ip_blacklist")
    op.drop_table("api_ip_blacklist")
