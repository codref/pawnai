"""Extract durable person facts from a transcript (not meeting action items).

Coworker ``extract_items`` owns decisions/commitments. This module only
captures biography-style knowledge that belongs on a People/ note.
"""

from __future__ import annotations

import logging
from typing import Any

from pawn_agent.core.coworker.jsonutil import parse_json_value
from pawn_agent.core.llm_sub import run as llm_run
from pawn_agent.utils.config import AgentConfig

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You extract durable facts about named people from a transcript. "
    "Reply with JSON only. Do not invent names, roles, or quotes. "
    "Skip meeting action items — those are handled elsewhere."
)

_PROMPT = """\
Known people in this session (gallery id → display name):
{roster}

Extract durable knowledge about those people only. Return JSON:
{{"people": [{{"speaker_id": "gallery-id",
"facts": ["one durable sentence about this person"],
"aliases": ["optional alternate name"],
"tags": ["optional topical tag without #"],
"summary": "optional short bio rewrite, or empty"}}]}}

Rules:
- Only include speaker_id values from the roster above.
- Facts are lasting: role, relationship, preference, expertise, how they work.
- Do NOT include one-off meeting tasks or decisions (e.g. "will send the doc").
- quote-backed facts preferred; leave facts empty when nothing durable was said.
- summary only when you can improve a one-paragraph card; else empty string.
- aliases only when the transcript clearly uses another name for them.
- tags are short topical labels (hiring, family, vendor). Never invent private data.
- Empty people list is fine.

TRANSCRIPT:
{source}
"""


async def extract_people_facts(
    cfg: AgentConfig,
    source_text: str,
    roster: dict[str, str],
) -> list[dict[str, Any]]:
    """Return per-person update dicts. Bad model output → empty list.

    Each dict: ``speaker_id``, ``facts``, ``aliases``, ``tags``, ``summary``.
    """
    if not (source_text or "").strip() or not roster:
        return []
    roster_lines = "\n".join(f"- {sid}: {name}" for sid, name in sorted(roster.items()))
    try:
        raw = await llm_run(
            cfg,
            _PROMPT.format(roster=roster_lines, source=source_text[:24000]),
            system_prompt=_SYSTEM,
        )
    except Exception as exc:
        logger.error("people extract failed: %s", exc, exc_info=True)
        raise
    return _normalize_people_payload(raw, set(roster))


def _normalize_people_payload(raw: str, allowed_ids: set[str]) -> list[dict[str, Any]]:
    try:
        data = parse_json_value(raw)
    except (ValueError, TypeError) as exc:
        logger.warning("people JSON parse failed: %s", exc)
        return []
    if isinstance(data, dict):
        data = data.get("people") or data.get("items") or []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in data:
        if not isinstance(entry, dict):
            continue
        sid = str(entry.get("speaker_id") or "").strip()
        if sid not in allowed_ids:
            continue
        facts = [
            str(f).strip()
            for f in (entry.get("facts") or [])
            if isinstance(f, (str, int, float)) and str(f).strip()
        ]
        aliases = [
            str(a).strip()
            for a in (entry.get("aliases") or [])
            if isinstance(a, (str, int, float)) and str(a).strip()
        ]
        tags = [
            str(t).strip().lstrip("#")
            for t in (entry.get("tags") or [])
            if isinstance(t, (str, int, float)) and str(t).strip()
        ]
        summary = str(entry.get("summary") or "").strip()
        if not facts and not aliases and not tags and not summary:
            continue
        out.append(
            {
                "speaker_id": sid,
                "facts": facts,
                "aliases": aliases,
                "tags": tags,
                "summary": summary,
            }
        )
    return out
