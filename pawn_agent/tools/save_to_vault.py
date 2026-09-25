"""Save Markdown / stored analysis to the S3 Obsidian vault."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pawn_agent.utils.config import AgentConfig
from pawn_core.vault import dump_frontmatter, resolve_path_template
from pawn_core.vault_config import vault_store_from_config


def format_analysis_markdown(row) -> str:
    """Render a SessionAnalysis ORM row as Markdown suitable for the vault."""
    parts: list[str] = []
    if row.title:
        parts.append(f"# {row.title}")
    if row.summary:
        parts.append(f"## Summary\n\n{row.summary}")
    if row.key_topics:
        parts.append(f"## Key Topics / Keywords\n\n{row.key_topics}")
    if row.speaker_highlights:
        parts.append(f"## Speaker Highlights\n\n{row.speaker_highlights}")
    if row.sentiment:
        parts.append(f"## Sentiment\n\n{row.sentiment}")
    if row.sentiment_tags:
        parts.append(f"## Sentiment Tags\n\n{', '.join(row.sentiment_tags)}")
    if row.tags:
        parts.append(f"## Tags\n\n{', '.join(row.tags)}")
    return "\n\n".join(parts).strip()


def save_to_vault_impl(
    cfg: AgentConfig,
    session_id: str,
    content: str,
    title: Optional[str] = None,
    path: Optional[str] = None,
    tags: Optional[list] = None,
) -> str:
    """Write analysis or free-form Markdown to the vault."""
    vault = cfg.vault
    if not vault.bucket:
        return "Error: vault.s3.bucket is not configured."
    doc_title = title or session_id
    when = datetime.now(timezone.utc)
    note_key = path or resolve_path_template(
        vault.analysis_path_template,
        agent_root=vault.agent_root,
        session_id=session_id,
        title=doc_title,
        now=when,
    )
    tag_list = list(tags or [])
    meta = {
        "pawn": "analysis",
        "session_id": session_id,
        "title": doc_title,
        "date": when.strftime("%Y-%m-%d"),
        "tags": tag_list or ["pawn/analysis"],
    }
    body = content if content.endswith("\n") else content + "\n"
    markdown = dump_frontmatter(meta, body)
    try:
        store = vault_store_from_config(cfg)
        store.write(note_key, markdown)
        return f"Saved analysis to vault: {note_key}"
    except Exception as exc:
        return f"Error saving to vault: {exc}"


def save_analysis_to_vault_impl(
    cfg: AgentConfig,
    session_id: str,
    *,
    title: Optional[str] = None,
    path: Optional[str] = None,
) -> str:
    """Load the latest DB analysis for *session_id* and save it to the vault."""
    from pawn_agent.utils.db import get_session_analysis  # noqa: PLC0415

    row = get_session_analysis(session_id, cfg.db_dsn)
    if row is None:
        return (
            f"Error: no stored analysis for session {session_id!r}. "
            "Run session_analyze first, or session_analyze --save."
        )
    content = format_analysis_markdown(row)
    if not content:
        return f"Error: analysis for {session_id!r} has no content fields to save."
    doc_title = title or row.title or session_id
    all_tags = list(row.tags or []) + list(row.sentiment_tags or [])
    return save_to_vault_impl(
        cfg,
        session_id=session_id,
        content=content,
        title=doc_title,
        path=path,
        tags=all_tags or None,
    )
