"""Vault note ledger, knowledge chunks, and item recurrence.

Revision ID: 0017
Revises: 0016
Create Date: 2026-09-27
"""

from __future__ import annotations

import os
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from pgvector.sqlalchemy import Vector

revision: str = "0017"
down_revision: Union[str, None] = "0016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_DIM: int = int(os.environ.get("PAWN_EMBED_DIM", "1024"))


def upgrade() -> None:
    op.add_column(
        "coworker_items",
        sa.Column("recurrence", sa.Integer(), nullable=False, server_default="0"),
    )
    op.create_table(
        "vault_note_state",
        sa.Column("key", sa.String(), primary_key=True),
        sa.Column("etag", sa.String(), nullable=True),
        sa.Column("content_hash", sa.String(), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(), nullable=True),
        sa.Column("last_processed_at", sa.DateTime(), nullable=True),
        sa.Column("last_processed_hash", sa.String(), nullable=True),
    )
    op.create_table(
        "knowledge_chunks",
        sa.Column("id", sa.String(), primary_key=True),
        sa.Column("source_kind", sa.String(), nullable=False),
        sa.Column("source_ref", sa.String(), nullable=False),
        sa.Column("heading", sa.String(), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(_DIM), nullable=False),
        sa.Column("content_hash", sa.String(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=True),
    )
    op.create_index("ix_knowledge_chunks_source_kind", "knowledge_chunks", ["source_kind"])
    op.create_index("ix_knowledge_chunks_source_ref", "knowledge_chunks", ["source_ref"])
    op.create_index("ix_knowledge_chunks_content_hash", "knowledge_chunks", ["content_hash"])
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_knowledge_chunks_embedding_hnsw "
        "ON knowledge_chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_knowledge_chunks_embedding_hnsw")
    op.drop_index("ix_knowledge_chunks_content_hash", table_name="knowledge_chunks")
    op.drop_index("ix_knowledge_chunks_source_ref", table_name="knowledge_chunks")
    op.drop_index("ix_knowledge_chunks_source_kind", table_name="knowledge_chunks")
    op.drop_table("knowledge_chunks")
    op.drop_table("vault_note_state")
    op.drop_column("coworker_items", "recurrence")
