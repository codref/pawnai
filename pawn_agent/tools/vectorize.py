"""Tool: vectorize — embed a session or SiYuan page into the knowledge store."""

from __future__ import annotations

from typing import Optional

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig

NAME = "vectorize"
DESCRIPTION = (
    "Embed content into the knowledge vector store so it can be searched semantically. "
    "Pass a session_id to index a session analysis, or a siyuan_path to index a SiYuan "
    "page. Idempotent — re-running replaces existing chunks for the same source."
)


async def vectorize_impl(
    cfg: AgentConfig,
    session_id: Optional[str] = None,
    siyuan_path: Optional[str] = None,
) -> str:
    """Core vectorize logic, callable from LangGraph tool nodes."""
    from pawn_agent.core.store import NS_SESSIONS, NS_SIYUAN, get_store  # noqa: PLC0415

    if not session_id and not siyuan_path:
        return "Error: provide either session_id or siyuan_path."
    if session_id and siyuan_path:
        return "Error: provide only one of session_id or siyuan_path."

    store = await get_store(cfg)

    if session_id:
        try:
            from pawn_agent.utils.db import get_session_analysis  # noqa: PLC0415
            from pawn_agent.utils.analysis import run_analysis  # noqa: PLC0415

            ran_analysis = False
            analysis = get_session_analysis(session_id, cfg.db_dsn)
            if analysis is None:
                await run_analysis(cfg, session_id)
                ran_analysis = True
                analysis = get_session_analysis(session_id, cfg.db_dsn)

            if analysis is None:
                return f"Error: could not produce analysis for session '{session_id}'."

            chunks: list[dict] = []

            # Chunk 1: overview
            overview_parts = []
            if analysis.title:
                overview_parts.append(f"Title: {analysis.title}")
            if analysis.summary:
                overview_parts.append(f"Summary: {analysis.summary}")
            if analysis.key_topics:
                overview_parts.append(f"Key topics:\n{analysis.key_topics}")
            if analysis.tags:
                overview_parts.append(f"Tags: {', '.join(analysis.tags)}")
            if overview_parts:
                chunks.append({
                    "key": f"{session_id}:overview",
                    "value": {
                        "text": "\n".join(overview_parts),
                        "session_id": session_id,
                        "title": analysis.title or session_id,
                        "chunk_type": "overview",
                        "tags": list(analysis.tags or []),
                    },
                })

            # Chunk 2: speaker highlights
            if analysis.speaker_highlights:
                chunks.append({
                    "key": f"{session_id}:highlights",
                    "value": {
                        "text": f"Speaker highlights:\n{analysis.speaker_highlights}",
                        "session_id": session_id,
                        "title": analysis.title or session_id,
                        "chunk_type": "highlights",
                    },
                })

            # Chunk 3: sentiment
            if analysis.sentiment:
                chunks.append({
                    "key": f"{session_id}:sentiment",
                    "value": {
                        "text": f"Sentiment: {analysis.sentiment}",
                        "session_id": session_id,
                        "title": analysis.title or session_id,
                        "chunk_type": "sentiment",
                    },
                })

            if not chunks:
                return f"Stored analysis for '{session_id}' has no embeddable content."

            for chunk in chunks:
                await store.aput(NS_SESSIONS, chunk["key"], chunk["value"])

            prefix = "Ran analysis then indexed" if ran_analysis else "Indexed"
            return f"{prefix} {len(chunks)} chunks for session '{session_id}'."

        except Exception as exc:
            return f"Error vectorizing session: {exc}"

    # siyuan_path branch
    notebook = cfg.siyuan_notebook
    if not notebook:
        return "Error: siyuan.notebook is not configured."
    try:
        from pawn_agent.utils.siyuan import siyuan_post  # noqa: PLC0415

        ids = siyuan_post(
            cfg.siyuan_url,
            cfg.siyuan_token,
            "/api/filetree/getIDsByHPath",
            {"path": siyuan_path, "notebook": notebook},
        )
    except Exception as exc:
        return f"Error resolving SiYuan path '{siyuan_path}': {exc}"

    if not isinstance(ids, list) or not ids:
        return f"No SiYuan page found at path '{siyuan_path}'."

    page_id = ids[0]
    try:
        from pawn_agent.utils.siyuan import siyuan_post  # noqa: PLC0415
        import re  # noqa: PLC0415

        _CONTENT_BLOCK_TYPES = {"p", "h", "h1", "h2", "h3", "h4", "h5", "h6"}
        _MD_STRIP_RE = re.compile(r"[*_`#\[\]()>~\-]+")
        _MIN_CHUNK_CHARS = 20

        try:
            info = siyuan_post(
                cfg.siyuan_url, cfg.siyuan_token,
                "/api/block/getBlockInfo", {"id": page_id},
            )
        except Exception:
            info = {}
        page_title = info.get("rootTitle") or page_id

        try:
            raw_blocks = siyuan_post(
                cfg.siyuan_url, cfg.siyuan_token,
                "/api/block/getChildBlocks", {"id": page_id},
            )
            if not isinstance(raw_blocks, list):
                raw_blocks = []
        except Exception:
            raw_blocks = []

        count = 0
        for blk in raw_blocks:
            if blk.get("type", "") not in _CONTENT_BLOCK_TYPES:
                continue
            text = _MD_STRIP_RE.sub(" ", blk.get("markdown") or blk.get("content") or "").strip()
            if len(text) < _MIN_CHUNK_CHARS:
                continue
            block_id = blk.get("id", "")
            await store.aput(
                NS_SIYUAN,
                f"{page_id}:{block_id}",
                {
                    "text": text,
                    "page_id": page_id,
                    "block_id": block_id,
                    "page_title": page_title,
                    "siyuan_path": siyuan_path,
                },
            )
            count += 1

        if count == 0:
            return f"No usable content blocks found for SiYuan page '{siyuan_path}'."
        return f"Indexed {count} chunks for SiYuan page '{siyuan_path}' (id: {page_id})."

    except Exception as exc:
        return f"Error vectorizing SiYuan page: {exc}"


def build(cfg: AgentConfig) -> Tool:
    async def vectorize(
        session_id: Optional[str] = None,
        siyuan_path: Optional[str] = None,
    ) -> str:
        """Embed content into the knowledge vector store.

        Provide exactly one of:
        - session_id: indexes the session analysis (overview, speaker highlights,
          sentiment). Runs analysis automatically if not yet stored.
        - siyuan_path: resolves the path to a SiYuan page and indexes its
          content blocks. The path must start with '/' (e.g. '/Notes/MyPage').

        Idempotent — re-running overwrites existing chunks for the same source.

        Args:
            session_id: Session identifier stored in the database.
            siyuan_path: Human-readable SiYuan document path.
        """
        return await vectorize_impl(cfg, session_id=session_id, siyuan_path=siyuan_path)

    return Tool(vectorize)
