"""Extract structured items from a transcript or note."""

from __future__ import annotations

import logging
from typing import Any

from pawn_agent.core.coworker.jsonutil import normalize_kind, parse_items_payload
from pawn_agent.core.llm_sub import run as llm_run
from pawn_agent.utils.config import AgentConfig

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You extract concrete items from a transcript or note. "
    "Reply with JSON only. Do not invent owners, dates, or quotes."
)

_PROMPT = """\
Extract items from the source below. Return JSON:
{{"items": [{{"kind": "decision|commitment|open_question|block|contradiction",
"text": "one sentence", "owner": "or empty", "due": "YYYY-MM-DD or empty",
"quote": "verbatim span from the source"}}]}}

Rules:
- A decision is a choice that was made.
- A commitment is something a person said they will do.
- An open_question is unresolved.
- A block is something that stops progress.
- A contradiction is a statement that conflicts with an earlier one in this source.
- Skip small talk, status with no decision, and tooling chatter.
- quote must be copied from the source. Use an empty list when nothing qualifies.

SOURCE:
{source}
"""


async def extract_items(cfg: AgentConfig, source_text: str) -> list[dict[str, Any]]:
    """Return normalized item dicts. Bad model output yields an empty list."""
    if not (source_text or "").strip():
        return []
    try:
        raw = await llm_run(cfg, _PROMPT.format(source=source_text[:24000]), system_prompt=_SYSTEM)
    except Exception as exc:
        logger.error("coworker extract failed: %s", exc, exc_info=True)
        raise
    items: list[dict[str, Any]] = []
    for entry in parse_items_payload(raw):
        text = str(entry.get("text") or "").strip()
        if not text:
            continue
        items.append(
            {
                "kind": normalize_kind(entry.get("kind")),
                "text": text,
                "owner": str(entry.get("owner") or "").strip(),
                "due": str(entry.get("due") or "").strip(),
                "quote": str(entry.get("quote") or "").strip(),
            }
        )
    return items
