"""Initial schema — all tables up to (but not including) session_analysis tags.

Creates the pgvector extension and the five core tables:
  - embeddings
  - speaker_names
  - transcription_segments
  - session_state
  - session_analysis  (title / summary / key_topics / speaker_highlights /
                       sentiment — without the tags columns added in 0002)

This migration is safe to run against an empty database **or** a database that
was bootstrapped via ``init_db()`` before Alembic was introduced.  For the
latter, stamp the existing database first so Alembic knows it is already at
this revision::

    alembic stamp 0001

Then apply only the subsequent deltas::

    alembic upgrade head

Revision ID: 0001
Revises:
Create Date: 2026-02-28
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector

# revision identifiers
revision: str = "0001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── pgvector extension ────────────────────────────────────────────────────
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # ── embeddings ────────────────────────────────────────────────────────────
    op.create_table(
        "embeddings",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("audio_file", sa.String(), nullable=False),
        sa.Column("local_speaker_label", sa.String(), nullable=False),
        sa.Column("start_time", sa.Float(), nullable=False),
        sa.Column("end_time", sa.Float(), nullable=False),
        sa.Column("embedding", Vector(512), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )

    # ── speaker_names ─────────────────────────────────────────────────────────
    op.create_table(
        "speaker_names",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("audio_file", sa.String(), nullable=False),
        sa.Column("local_speaker_label", sa.String(), nullable=False),
        sa.Column("speaker_name", sa.String(), nullable=False),
        sa.Column("labeled_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )

    # ── transcription_segments ────────────────────────────────────────────────
    op.create_table(
        "transcription_segments",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("audio_file", sa.String(), nullable=False),
        sa.Column("original_speaker_label", sa.String(), nullable=True),
        sa.Column("start_time", sa.Float(), nullable=False),
        sa.Column("end_time", sa.Float(), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("words", JSONB(), nullable=True),
        sa.Column("segment_index", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index(
        "ix_transcription_segments_session_id",
        "transcription_segments",
        ["session_id"],
        if_not_exists=True,
    )

    # ── session_state ─────────────────────────────────────────────────────────
    op.create_table(
        "session_state",
        sa.Column("session_id", sa.String(), nullable=False),
        sa.Column("processed_files", JSONB(), nullable=True),
        sa.Column("speaker_embeddings", JSONB(), nullable=True),
        sa.Column("time_cursor", sa.Float(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("session_id"),
        if_not_exists=True,
    )

    # ── session_analysis (pre-tags) ───────────────────────────────────────────
    op.create_table(
        "session_analysis",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("session_id", sa.String(), nullable=True),
        sa.Column("source", sa.String(), nullable=False),
        sa.Column("model", sa.String(), nullable=False),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("summary", sa.Text(), nullable=True),
        sa.Column("key_topics", sa.Text(), nullable=True),
        sa.Column("speaker_highlights", sa.Text(), nullable=True),
        sa.Column("sentiment", sa.Text(), nullable=True),
        sa.Column("analyzed_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index(
        "ix_session_analysis_session_id",
        "session_analysis",
        ["session_id"],
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_session_analysis_session_id", table_name="session_analysis")
    op.drop_table("session_analysis")
    op.drop_table("session_state")
    op.drop_index("ix_transcription_segments_session_id", table_name="transcription_segments")
    op.drop_table("transcription_segments")
    op.drop_table("speaker_names")
    op.drop_table("embeddings")
    op.execute("DROP EXTENSION IF EXISTS vector")
