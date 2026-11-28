"""Portal brainmap boards: per-board card positions and viewport.

Revision ID: 0033
Revises: 0032
Create Date: 2026-11-28

See docs/PORTAL.md "Brainmap". One row per board id (``items`` first);
``positions`` maps node ids to ``{x, y, pinned}``.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0033"
down_revision: Union[str, None] = "0032"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "portal_boards",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("name", sa.String(), nullable=False),
        sa.Column("positions", sa.JSON(), nullable=True),
        sa.Column("viewport", sa.JSON(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("name", name="uq_portal_boards_name"),
    )


def downgrade() -> None:
    op.drop_table("portal_boards")
