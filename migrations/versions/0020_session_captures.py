"""Notes and screenshots attached to a diarization session.

Revision ID: 0020
Revises: 0019
Create Date: 2026-10-01
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0020"
down_revision: Union[str, None] = "0019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "session_captures",
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("item_id", sa.String(), nullable=False),
        sa.Column("kind", sa.String(), nullable=False),
        sa.Column("at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("at_offset_minutes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("text", sa.Text(), nullable=True),
        sa.Column("s3_uri", sa.Text(), nullable=True),
        sa.Column("output", sa.String(), nullable=True),
        sa.Column("region", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column("vault_key", sa.String(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("audio_offset_s", sa.Float(), nullable=True),
        sa.Column("chunk_audio_start", sa.Float(), nullable=False, server_default="0"),
        sa.Column("received_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("session_id", "item_id"),
    )


def downgrade() -> None:
    op.drop_table("session_captures")
