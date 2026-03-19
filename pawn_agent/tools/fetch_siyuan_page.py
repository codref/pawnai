"""Tool: fetch_siyuan_page — retrieve a SiYuan page's content by path."""

from __future__ import annotations

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.siyuan import siyuan_post

NAME = "fetch_siyuan_page"
DESCRIPTION = (
    "Fetch the text content of a SiYuan page given its human-readable path "
    "(e.g. '/Notes/MyPage'). Returns the page content so you can read, analyse, "
    "or ask the user whether to vectorize it."
)


def build(cfg: AgentConfig) -> Tool:
    def fetch_siyuan_page(path: str) -> str:
        """Retrieve the content of a SiYuan document by its path.

        Resolves the path to a document ID, then fetches all content blocks
        and returns them as plain text. Use this to read a page before
        analysing or vectorizing it.

        Args:
            path: Human-readable document path starting with '/'
                  (e.g. '/Conversations/2026-03/standup').
        """
        notebook = cfg.siyuan_notebook
        if not notebook:
            return "Error: siyuan.notebook is not configured."

        try:
            ids = siyuan_post(
                cfg.siyuan_url, cfg.siyuan_token,
                "/api/filetree/getIDsByHPath",
                {"path": path, "notebook": notebook},
            )
        except Exception as exc:
            return f"Error resolving path '{path}': {exc}"

        if not isinstance(ids, list) or not ids:
            return f"No SiYuan page found at path '{path}'."

        page_id = ids[0]

        try:
            info = siyuan_post(
                cfg.siyuan_url, cfg.siyuan_token,
                "/api/block/getBlockInfo",
                {"id": page_id},
            )
            page_title = (info or {}).get("rootTitle") or path
        except Exception:
            page_title = path

        try:
            blocks = siyuan_post(
                cfg.siyuan_url, cfg.siyuan_token,
                "/api/block/getChildBlocks",
                {"id": page_id},
            )
            if not isinstance(blocks, list):
                blocks = []
        except Exception as exc:
            return f"Error fetching blocks for page '{path}': {exc}"

        lines = [f"# {page_title}", f"*(page_id: {page_id})*", ""]
        for blk in blocks:
            text = (blk.get("markdown") or blk.get("content") or "").strip()
            if text:
                lines.append(text)

        if len(lines) <= 3:
            return f"Page '{path}' exists (id: {page_id}) but has no readable content."

        return "\n\n".join(lines)

    return Tool(fetch_siyuan_page)
