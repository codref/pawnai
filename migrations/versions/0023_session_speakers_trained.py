"""Add speakers_trained_at on session_state.

Revision ID: 0023
Revises: 0022
Create Date: 2026-10-13

Lets the user mark a session as speaker-trained so leftover noise
SPEAKER_XX labels do not keep the Sessions ``needs training`` badge.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "0023"
down_revision: Union[str, None] = "0022"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "session_state",
        sa.Column("speakers_trained_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("session_state", "speakers_trained_at")
