"""Extract structured items from a transcript or note."""

from __future__ import annotations

import logging
from typing import Any

from pawn_agent.core.coworker.jsonutil import (
    normalize_kind,
    parse_items_payload,
)
from pawn_agent.core.llm_sub import run as llm_run
from pawn_agent.utils.config import AgentConfig

logger = logging.getLogger(__name__)

# Soft caps after the model replies (characters).
_CONTEXT_MAX = 400
_QUOTE_MAX = 800

_SYSTEM = (
    "You extract concrete items from a transcript or note. "
    "Reply with JSON only. Do not invent owners, dates, "
    "quotes, or context."
)

_PROMPT = """\
Extract items from the source below. Return JSON:
{{"items": [{{"kind": "decision|commitment|open_question|block|contradiction",
"text": "one-line claim", "context": "2-4 self-contained sentences",
"owner": "or empty", "due": "YYYY-MM-DD or empty",
"quote": "verbatim supporting passage from the source"}}]}}

Rules:
- A decision is a choice that was made.
- A commitment is something a person said they will do.
- An open_question is unresolved.
- A block is something that stops progress.
- A contradiction is a statement that conflicts with an
  earlier one in this source.
- Skip small talk, status with no decision, and tooling chatter.
- Prefer fewer richer items over many thin ones. A stranger
  should triage each item without reopening the source.
- text is a short actionable claim (one sentence).
- context must answer who, about what, what changed, and
  what is at stake. Repeat situational facts across sibling
  items; do not factor a shared preamble. Only facts from
  SOURCE; if unknown, say so briefly (e.g. owner unclear).
- quote must be copied from the source as the supporting
  passage (several lines of dialogue or prose when available).
  Prefer dropping an item over inventing a quote or using an
  empty/weak quote that only restates text.
- Use an empty list when nothing qualifies.

SOURCE:
{source}
"""


def _soft_trim(value: str, limit: int) -> str:
    text = (value or "").strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip()
    return cut or text[:limit]


def normalize_extracted_entry(
    entry: dict[str, Any],
) -> dict[str, Any] | None:
    """Normalize one raw model item. None when claim empty."""
    text = str(entry.get("text") or "").strip()
    if not text:
        return None
    quote = _soft_trim(str(entry.get("quote") or ""), _QUOTE_MAX)
    context = _soft_trim(str(entry.get("context") or ""), _CONTEXT_MAX)
    return {
        "kind": normalize_kind(entry.get("kind")),
        "text": text,
        "context": context,
        "owner": str(entry.get("owner") or "").strip(),
        "due": str(entry.get("due") or "").strip(),
        "quote": quote,
    }


async def extract_items(
    cfg: AgentConfig, source_text: str
) -> list[dict[str, Any]]:
    """Return normalized item dicts. Bad output → empty list."""
    if not (source_text or "").strip():
        return []
    try:
        prompt = _PROMPT.format(source=source_text[:24000])
        raw = await llm_run(cfg, prompt, system_prompt=_SYSTEM)
    except Exception as exc:
        logger.error("coworker extract failed: %s", exc, exc_info=True)
        raise
    items: list[dict[str, Any]] = []
    for entry in parse_items_payload(raw):
        normalized = normalize_extracted_entry(entry)
        if normalized is None:
            continue
        items.append(normalized)
    return items
