"""LLM meeting prep — search sessions/items/people/vault, write ## Prep."""

from __future__ import annotations

import asyncio
import logging
from typing import Any
from uuid import uuid4

from pawn_agent.core.agent_runner import run_agent_turn
from pawn_agent.core.sallm_registry import SallmSessionRegistry
from pawn_core.vault import VaultNotFound, normalize_vault_key, parse_frontmatter
from pawn_core.vault_config import vault_store_from_config
from pawn_server.core.planner import (
    PlannerError,
    find_event_by_id,
    replace_prep_section,
)
from pawn_server.core.vault_events import publish_vault_event

logger = logging.getLogger(__name__)


def build_prep_prompt(cfg: Any, *, path: str, event: dict[str, Any]) -> str:
    """Fixed brief for the ``meeting_prep`` skill."""
    goals_path = str(
        getattr(getattr(cfg, "coworker", None), "goals_path", None) or "Goals.md"
    )
    lines = [
        "Active skill: meeting_prep.",
        "Prepare the user for this planner event. Search Pawn memory with tools;",
        "do not invent facts. Write only the ## Prep section via event_prep_update.",
        "",
        f"Event note path: {path}",
        f"Event id: {event.get('id') or ''}",
        f"Title: {event.get('title') or ''}",
        f"Start: {event.get('start') or ''}",
        f"End: {event.get('end') or ''}",
    ]
    attendees = event.get("attendees") or []
    if attendees:
        lines.append("Attendees: " + ", ".join(str(a) for a in attendees))
    topics = event.get("topics") or []
    if topics:
        lines.append("Topics: " + ", ".join(str(t) for t in topics))
    if event.get("session_id"):
        lines.append(f"Linked session_id: {event['session_id']}")
    if event.get("source_url"):
        lines.append(f"Source URL: {event['source_url']}")
    body = (event.get("body") or "").strip()
    if body:
        lines.extend(["", "Event body:", body[:2000]])
    lines.extend(
        [
            "",
            f"Optional: note_read --path {goals_path} for Active threads.",
            "Steps:",
            "1) note_read the event note.",
            "2) sessions_list / session_transcript for related past meetings "
            "(title, attendees, topics). Prefer the linked session_id when set.",
            "3) people_show for named attendees when useful.",
            "4) note_search / knowledge_search for vault notes and open loops.",
            "5) event_prep_update --event-id <id> --content-file @prep with a "
            "```file prep block: sparse bullets + [[wikilinks]]. Skip filler.",
            "6) Answer with one line: what you wrote and how many links.",
            "Never dump tool errors into the vault. Never rewrite the event body "
            "outside ## Prep.",
        ]
    )
    return "\n".join(lines)


async def run_meeting_prep(
    cfg: Any,
    *,
    event_id: str,
    path: str = "",
    registry: SallmSessionRegistry,
    store: Any = None,
    job_id: str = "",
) -> str:
    """Run one prep turn. Returns the agent reply text."""
    if store is None:
        store = vault_store_from_config(cfg)
    ev = await asyncio.to_thread(find_event_by_id, cfg, event_id, store=store)
    key = normalize_vault_key(path or ev.get("path") or "")
    if not key:
        raise FileNotFoundError(f"event path missing for {event_id}")
    if not key.lower().endswith(".md"):
        key = f"{key}.md"
    try:
        note_text = await asyncio.to_thread(store.read, key)
    except VaultNotFound as exc:
        raise FileNotFoundError(f"event note not found: {key}") from exc
    meta, _ = parse_frontmatter(note_text)
    if str(meta.get("pawn") or "").strip() != "event":
        raise PlannerError(f"not a planner event: {key}", status_code=422)

    prompt = build_prep_prompt(cfg, path=key, event=ev)
    conv = f"note:{key}"
    result = await run_agent_turn(
        cfg=cfg,
        registry=registry,
        prompt=prompt,
        session_id=conv,
        source="planner",
        command="meeting_prep",
    )
    # If the agent already wrote ## Prep via the tool, leave it. Otherwise
    # treat a non-empty reply without a tool write as best-effort content only
    # when ## Prep is still empty (tool path is preferred).
    try:
        after = await asyncio.to_thread(store.read, key)
        from pawn_server.core.planner import extract_prep  # noqa: PLC0415

        if not extract_prep(after) and (result.response or "").strip():
            from pawn_agent.core.sallm_session import strip_tool_trail  # noqa: PLC0415

            display = strip_tool_trail(result.response or "").strip()
            if display and not display.lower().startswith("error"):
                updated = replace_prep_section(after, display, prep_job_id=job_id)
                await asyncio.to_thread(store.write, key, updated)
                publish_vault_event([key], source="planner")
        elif job_id:
            # Stamp prep_job_id if tool wrote prep without the job id.
            meta2, body2 = parse_frontmatter(after)
            if str(meta2.get("prep_job_id") or "") != job_id and extract_prep(after):
                meta2["prep_job_id"] = job_id
                from pawn_core.vault import dump_frontmatter  # noqa: PLC0415

                stamped = dump_frontmatter(meta2, body2)
                # refresh prep_at via replace if missing
                if not meta2.get("prep_at"):
                    stamped = replace_prep_section(
                        after, extract_prep(after), prep_job_id=job_id
                    )
                await asyncio.to_thread(store.write, key, stamped)
                publish_vault_event([key], source="planner")
    except Exception as exc:  # noqa: BLE001
        logger.debug("meeting_prep post-write check: %s", exc)

    return result.response or ""


def new_prep_job_id() -> str:
    return str(uuid4())
