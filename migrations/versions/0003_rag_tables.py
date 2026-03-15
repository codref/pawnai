"""Add rag_sources and text_chunks tables for RAG vector search.

Creates two tables:
  - rag_sources  — one row per indexed source (transcript session, SiYuan page,
                   or any future source type); extensible without schema changes.
  - text_chunks  — text chunks with sentence-transformer embeddings, referenced
                   via source_id FK to rag_sources.

The embedding dimension defaults to 2048 (Qwen3-Embedding-0.6B) but can be
overridden by setting ``PAWN_EMBED_DIM`` before running the migration::

    PAWN_EMBED_DIM=384 alembic upgrade head

This value must match ``TEXT_CHUNK_DIM`` in ``database.py`` and the
``rag.embed_dim`` setting in ``pawnai.yaml``.

Revision ID: 0003
Revises: 0002
Create Date: 2026-03-15
"""

from __future__ import annotations

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0003"
down_revision: Union[str, None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_DIM: int = int(os.environ.get("PAWN_EMBED_DIM", "1024"))


def upgrade() -> None:
    op.create_table(
        "rag_sources",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("external_id", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column("metadata", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        if_not_exists=True,
    )
    op.create_index(
        "ix_rag_sources_source_type",
        "rag_sources",
        ["source_type"],
        if_not_exists=True,
    )
    op.create_index(
        "ix_rag_sources_external_id",
        "rag_sources",
        ["external_id"],
        if_not_exists=True,
    )

    op.create_table(
        "text_chunks",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("source_id", sa.String(), nullable=False),
        sa.Column("speaker_name", sa.String(), nullable=True),
        sa.Column("start_time", sa.Float(), nullable=True),
        sa.Column("end_time", sa.Float(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(_DIM), nullable=False),
        sa.Column("metadata", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(["source_id"], ["rag_sources.id"], ondelete="CASCADE"),
        if_not_exists=True,
    )
    op.create_index(
        "ix_text_chunks_source_id",
        "text_chunks",
        ["source_id"],
        if_not_exists=True,
    )
    # HNSW index for approximate nearest-neighbour cosine search.
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_text_chunks_embedding_hnsw "
        "ON text_chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_text_chunks_embedding_hnsw")
    op.drop_index("ix_text_chunks_source_id", table_name="text_chunks")
    op.drop_table("text_chunks")
    op.drop_index("ix_rag_sources_external_id", table_name="rag_sources")
    op.drop_index("ix_rag_sources_source_type", table_name="rag_sources")
    op.drop_table("rag_sources")
