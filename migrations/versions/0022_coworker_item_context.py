"""Add situational context column on coworker items.

Revision ID: 0022
Revises: 0021
Create Date: 2026-10-12

Self-contained review text (who / about what / stakes) separate from the
one-line claim in ``text`` and the verbatim ``quote``.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0022"
down_revision: Union[str, None] = "0021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column("coworker_items", sa.Column("context", sa.Text(), nullable=True))


def downgrade() -> None:
    op.drop_column("coworker_items", "context")
