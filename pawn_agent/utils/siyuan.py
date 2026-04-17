"""SiYuan Notes helpers for pawn-agent."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pawn_agent.utils.config import AgentConfig
from pawn_core.siyuan import infer_title, resolve_path, siyuan_post  # noqa: F401


def build_siyuan_block_url(block_id: str) -> str:
    """Return a SiYuan deep link for a document/block id."""
    return f"siyuan://blocks/{block_id}"


def build_siyuan_markdown(sections: dict, session_id: str, transcript: str, model: str) -> str:
    title = sections.get("title") or "Conversation Analysis"
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    tags_str = ", ".join(sections.get("tags") or [])
    sentiment_str = ", ".join(sections.get("sentiment_tags") or [])

    header = f"**Session:** `{session_id}` · **Date:** {now_str} · **Model:** {model}"
    if tags_str:
        header += f"\n**Tags:** {tags_str}"
    if sentiment_str:
        header += f"\n**Sentiment tags:** {sentiment_str}"

    parts = [f"# {title}", header, "---"]
    for heading, key in [
        ("Summary", "summary"),
        ("Key Topics / Keywords", "key_topics"),
        ("Speaker Highlights", "speaker_highlights"),
        ("Sentiment", "sentiment"),
    ]:
        val = sections.get(key)
        if val:
            parts.append(f"## {heading}\n\n{val}")
    parts += ["---", f"## Transcript\n\n{transcript or '_No transcript._'}"]
    return "\n\n".join(parts)


def do_save_to_siyuan(
    cfg: AgentConfig,
    session_id: str,
    title: Optional[str],
    content: str,
    path: Optional[str],
    tags: Optional[list] = None,
) -> str:
    url = cfg.siyuan_url
    token = cfg.siyuan_token
    notebook = cfg.siyuan_notebook
    if not notebook:
        return "SiYuan notebook ID is not configured (set siyuan.notebook in config)."
    title = title or infer_title(content, session_id)
    resolved_path = path or resolve_path(cfg.siyuan_path_template, session_id, title)
    try:
        ids_data = siyuan_post(url, token, "/api/filetree/getIDsByHPath",
                               {"path": resolved_path, "notebook": notebook})
        for doc_id in (ids_data if isinstance(ids_data, list) else []):
            siyuan_post(url, token, "/api/filetree/removeDocByID", {"id": doc_id})
    except Exception:
        pass
    doc_id = siyuan_post(url, token, "/api/filetree/createDocWithMd",
                         {"notebook": notebook, "path": resolved_path, "markdown": content})
    if doc_id:
        try:
            attrs: dict = {"custom-session-id": session_id}
            if tags:
                attrs["tags"] = ",".join(tags)
            siyuan_post(url, token, "/api/attr/setBlockAttrs",
                        {"id": doc_id, "attrs": attrs})
        except Exception:
            pass
    try:
        daily_path = resolve_path(cfg.siyuan_daily_template, session_id, None)
        daily_ids = siyuan_post(url, token, "/api/filetree/getIDsByHPath",
                                {"path": daily_path, "notebook": notebook})
        if isinstance(daily_ids, list) and daily_ids:
            daily_doc_id = daily_ids[0]
        else:
            daily_doc_id = siyuan_post(url, token, "/api/filetree/createDocWithMd",
                                       {"notebook": notebook, "path": daily_path, "markdown": ""})
        if daily_doc_id and doc_id:
            siyuan_post(url, token, "/api/block/appendBlock",
                        {"dataType": "markdown", "data": f'(({doc_id} "{title}"))',
                         "parentID": daily_doc_id})
    except Exception:
        pass
    if doc_id:
        return build_siyuan_block_url(str(doc_id))
    return "Document created (no ID returned)."
