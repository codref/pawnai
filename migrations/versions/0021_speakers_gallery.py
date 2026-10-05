"""Curated Speakers gallery (people + enrollments + session maps).

Revision ID: 0021
Revises: 0020
Create Date: 2026-10-05

Replaces the old auto-accumulating ``embeddings`` soup as the source of
truth for *who is this person*.  Identity matching now uses only rows in
``speaker_enrollments`` that a human explicitly approved.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0021"
down_revision: Union[str, None] = "0020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "speakers",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=False),
        sa.Column(
            "aliases",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column(
            "active",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_speakers_display_name", "speakers", ["display_name"])

    # Embeddings are JSONB (list of floats), not pgvector.
    # A curated gallery is small (tens of speakers × a few enrollments), so
    # cosine scoring in Python stays readable and does not lock the schema
    # to one embedding dimension when models change.
    op.create_table(
        "speaker_enrollments",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("speaker_id", sa.String(), nullable=False),
        sa.Column(
            "embedding",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
        ),
        sa.Column("embedding_model", sa.String(), nullable=False),
        sa.Column("embedding_dim", sa.Integer(), nullable=False),
        sa.Column("source_session_id", sa.String(), nullable=True),
        sa.Column("source_audio_file", sa.String(), nullable=True),
        sa.Column("start_time", sa.Float(), nullable=True),
        sa.Column("end_time", sa.Float(), nullable=True),
        sa.Column("duration", sa.Float(), nullable=False, server_default="0"),
        sa.Column("quality_score", sa.Float(), nullable=True),
        sa.Column("notes", sa.Text(), nullable=True),
        sa.Column("approved_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["speaker_id"], ["speakers.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        "ix_speaker_enrollments_speaker_id",
        "speaker_enrollments",
        ["speaker_id"],
    )

    op.create_table(
        "session_speaker_map",
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("local_label", sa.String(), nullable=False),
        sa.Column("speaker_id", sa.String(), nullable=True),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column("match_score", sa.Float(), nullable=True),
        sa.Column(
            "match_method",
            sa.String(),
            nullable=False,
            server_default="manual",
        ),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.ForeignKeyConstraint(["speaker_id"], ["speakers.id"], ondelete="SET NULL"),
        sa.PrimaryKeyConstraint("session_id", "local_label"),
    )


def downgrade() -> None:
    op.drop_table("session_speaker_map")
    op.drop_index("ix_speaker_enrollments_speaker_id", table_name="speaker_enrollments")
    op.drop_table("speaker_enrollments")
    op.drop_index("ix_speakers_display_name", table_name="speakers")
    op.drop_table("speakers")
