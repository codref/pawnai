"""Add tags and sentiment_tags to session_analysis.

Adds two JSONB columns with GIN indexes to ``session_analysis`` to support
efficient tag-based search and grouping via PostgreSQL ``@>`` containment
queries:

  - ``tags``           — 5-10 topic/entity/tone tags (JSONB array)
  - ``sentiment_tags`` — up to 3 short sentiment labels (JSONB array)

Both columns are nullable so that existing rows (written before this migration)
are unaffected.

For databases that were created via ``init_db()`` *after* the tags feature was
already in the codebase (i.e. the table already has these columns), stamp the
database at this revision to mark it as current::

    alembic stamp 0002

Revision ID: 0002
Revises: 0001
Create Date: 2026-02-28
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import JSONB

# revision identifiers
revision: str = "0002"
down_revision: Union[str, None] = "0001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "session_analysis",
        sa.Column("sentiment_tags", JSONB(), nullable=True),
    )
    op.add_column(
        "session_analysis",
        sa.Column("tags", JSONB(), nullable=True),
    )
    op.create_index(
        "ix_session_analysis_sentiment_tags_gin",
        "session_analysis",
        ["sentiment_tags"],
        postgresql_using="gin",
        if_not_exists=True,
    )
    op.create_index(
        "ix_session_analysis_tags_gin",
        "session_analysis",
        ["tags"],
        postgresql_using="gin",
        if_not_exists=True,
    )


def downgrade() -> None:
    op.drop_index("ix_session_analysis_tags_gin", table_name="session_analysis")
    op.drop_index("ix_session_analysis_sentiment_tags_gin", table_name="session_analysis")
    op.drop_column("session_analysis", "tags")
    op.drop_column("session_analysis", "sentiment_tags")
