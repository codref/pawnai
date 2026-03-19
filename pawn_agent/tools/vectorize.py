"""Tool: vectorize — embed a session transcript or SiYuan page into the RAG index."""

from __future__ import annotations

from typing import Optional

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig

NAME = "vectorize"
DESCRIPTION = (
    "Embed content into the RAG vector index so it can be searched semantically. "
    "Pass a session_id to index a session, or a siyuan_path to index a SiYuan page. "
    "For sessions: uses the stored analysis (summary, topics, highlights) if available; "
    "falls back to raw transcript speaker turns if not. "
    "Never call get_analysis or query_conversation before this — it reads the DB directly. "
    "Idempotent — re-running replaces existing chunks for the same source."
)


def build(cfg: AgentConfig) -> Tool:
    async def vectorize(
        session_id: Optional[str] = None,
        siyuan_path: Optional[str] = None,
    ) -> str:
        """Embed content into the RAG vector index.

        Provide exactly one of:
        - session_id: indexes the full transcript using speaker-turn chunking,
          enriched with analysis tags if a session analysis exists.
        - siyuan_path: resolves the path to a SiYuan page and indexes its
          content blocks. The path must start with '/' (e.g. '/Notes/MyPage').

        The source reference is stored in the rag_sources table so it can be
        queried later.

        Args:
            session_id: Session identifier stored in the database.
            siyuan_path: Human-readable SiYuan document path.
        """
        from pawn_agent.utils.vectorize import vectorize_session, vectorize_siyuan_page
        from pawn_agent.utils.siyuan import siyuan_post

        if not session_id and not siyuan_path:
            return "Error: provide either session_id or siyuan_path."
        if session_id and siyuan_path:
            return "Error: provide only one of session_id or siyuan_path."

        if session_id:
            try:
                from pawn_agent.utils.db import get_session_analysis
                from pawn_agent.utils.analysis import run_analysis

                ran_analysis = False
                if get_session_analysis(session_id, cfg.db_dsn) is None:
                    await run_analysis(cfg, session_id)
                    ran_analysis = True

                n, _ = vectorize_session(
                    session_id=session_id,
                    db_dsn=cfg.db_dsn,
                    embed_model=cfg.embed_model,
                    embed_device=cfg.embed_device,
                    embed_dim=cfg.embed_dim,
                )
                prefix = "Ran analysis then indexed" if ran_analysis else "Indexed"
                return f"{prefix} {n} chunks for session '{session_id}'."
            except Exception as exc:
                return f"Error vectorizing session: {exc}"

        # siyuan_path branch — resolve path to page ID first
        notebook = cfg.siyuan_notebook
        if not notebook:
            return "Error: siyuan.notebook is not configured."
        try:
            ids = siyuan_post(
                cfg.siyuan_url, cfg.siyuan_token,
                "/api/filetree/getIDsByHPath",
                {"path": siyuan_path, "notebook": notebook},
            )
        except Exception as exc:
            return f"Error resolving SiYuan path '{siyuan_path}': {exc}"

        if not isinstance(ids, list) or not ids:
            return f"No SiYuan page found at path '{siyuan_path}'."

        page_id = ids[0]
        try:
            n = vectorize_siyuan_page(
                page_id=page_id,
                siyuan_url=cfg.siyuan_url,
                siyuan_token=cfg.siyuan_token,
                db_dsn=cfg.db_dsn,
                embed_model=cfg.embed_model,
                embed_device=cfg.embed_device,
                embed_dim=cfg.embed_dim,
            )
            return f"Indexed {n} chunks for SiYuan page '{siyuan_path}' (id: {page_id})."
        except Exception as exc:
            return f"Error vectorizing SiYuan page: {exc}"

    return Tool(vectorize)
