"""Session audio offsets and the reprocess marker.

Revision ID: 0028
Revises: 0027
Create Date: 2026-11-27

``session_state.file_offsets`` stores ``{path, start, end}`` per audio file so
the portal player can map a global timestamp onto ``(file index, local time)``.
``session_state.processing`` is set while a rediarize or retranscribe job is
queued or running, and cleared when the diarize worker finishes.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0028"
down_revision: Union[str, None] = "0027"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "session_state",
        sa.Column("file_offsets", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )
    op.add_column(
        "session_state",
        sa.Column("processing", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("session_state", "processing")
    op.drop_column("session_state", "file_offsets")
