"""Drop rag_sources and text_chunks tables (replaced by AsyncPostgresStore).

The RAG vector pipeline now uses LangGraph's AsyncPostgresStore which manages
its own ``store`` and ``store_vectors`` tables created via ``store.setup()``
at application startup.

The HNSW index is dropped first because it cannot be dropped implicitly by
``DROP TABLE`` when using ``IF EXISTS``.

Revision ID: 0009
Revises: 0008
Create Date: 2026-04-22
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

revision: str = "0009"
down_revision: Union[str, None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop HNSW index before dropping the table.
    op.execute("DROP INDEX IF EXISTS ix_text_chunks_embedding_hnsw")
    op.execute("DROP INDEX IF EXISTS ix_text_chunks_source_id")
    op.execute("DROP TABLE IF EXISTS text_chunks")
    op.execute("DROP INDEX IF EXISTS ix_rag_sources_external_id")
    op.execute("DROP INDEX IF EXISTS ix_rag_sources_source_type")
    op.execute("DROP TABLE IF EXISTS rag_sources")


def downgrade() -> None:
    """Restore rag_sources and text_chunks with a fixed 1024-dim embedding column."""
    import os

    from pgvector.sqlalchemy import Vector

    _DIM: int = int(os.environ.get("PAWN_EMBED_DIM", "1024"))

    op.create_table(
        "rag_sources",
        sa.Column("id", sa.String(), nullable=False),
        sa.Column("source_type", sa.String(), nullable=False),
        sa.Column("external_id", sa.String(), nullable=False),
        sa.Column("display_name", sa.String(), nullable=True),
        sa.Column("metadata", JSONB(), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index("ix_rag_sources_source_type", "rag_sources", ["source_type"])
    op.create_index("ix_rag_sources_external_id", "rag_sources", ["external_id"])

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
    )
    op.create_index("ix_text_chunks_source_id", "text_chunks", ["source_id"])
    op.execute(
        "CREATE INDEX ix_text_chunks_embedding_hnsw "
        "ON text_chunks USING hnsw (embedding vector_cosine_ops) "
        "WITH (m = 16, ef_construction = 64)"
    )
