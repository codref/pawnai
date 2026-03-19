"""Tool: rag_stats — display a summary of the RAG vector index."""

from __future__ import annotations

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig

NAME = "rag_stats"
DESCRIPTION = (
    "Show a summary of the RAG vector index: indexed sources and chunk counts "
    "by type, plus a per-source detail list with last-indexed timestamps."
)


def build(cfg: AgentConfig) -> Tool:
    def rag_stats() -> str:
        """Return a summary of all sources currently indexed in the RAG database.

        Shows total sources and chunks broken down by type (transcript, siyuan, etc.)
        and a per-source detail list ordered by most recently indexed.
        """
        from sqlalchemy import create_engine, func, select
        from sqlalchemy.orm import Session

        from pawn_agent.utils.db import RagSource, TextChunk

        engine = create_engine(cfg.db_dsn)

        with Session(engine) as db:
            summary_rows = db.execute(
                select(
                    RagSource.source_type,
                    func.count(RagSource.id.distinct()).label("sources"),
                    func.count(TextChunk.id).label("chunks"),
                )
                .outerjoin(TextChunk, TextChunk.source_id == RagSource.id)
                .group_by(RagSource.source_type)
                .order_by(RagSource.source_type)
            ).all()

            detail_rows = db.execute(
                select(
                    RagSource.source_type,
                    RagSource.display_name,
                    RagSource.external_id,
                    func.count(TextChunk.id).label("chunks"),
                    RagSource.created_at,
                )
                .outerjoin(TextChunk, TextChunk.source_id == RagSource.id)
                .group_by(RagSource.id)
                .order_by(RagSource.created_at.desc())
            ).all()

        if not summary_rows:
            return "No sources indexed yet. Use the vectorize tool to index a session or SiYuan page."

        total_sources = sum(r.sources for r in summary_rows)
        total_chunks = sum(r.chunks for r in summary_rows)

        lines = ["**RAG Index Summary**", ""]
        lines.append(f"{'Type':<12} {'Sources':>7} {'Chunks':>7}")
        lines.append("-" * 28)
        for r in summary_rows:
            lines.append(f"{r.source_type:<12} {r.sources:>7} {r.chunks:>7}")
        lines.append("-" * 28)
        lines.append(f"{'TOTAL':<12} {total_sources:>7} {total_chunks:>7}")
        lines.append("")
        lines.append("**Sources (most recent first)**")
        lines.append("")
        for r in detail_rows:
            indexed = r.created_at.strftime("%Y-%m-%d %H:%M") if r.created_at else "unknown"
            name = r.display_name or r.external_id
            lines.append(f"- [{r.source_type}] {name} — {r.chunks} chunks, indexed {indexed}")

        return "\n".join(lines)

    return Tool(rag_stats)
