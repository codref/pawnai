"""Score extracted items against active goal threads."""

from __future__ import annotations

import logging
from typing import Any

from pawn_agent.core.coworker.jsonutil import parse_items_payload
from pawn_agent.core.llm_sub import run as llm_run
from pawn_agent.utils.config import AgentConfig
from pawn_core.goals import GoalThread

logger = logging.getLogger(__name__)

_SYSTEM = (
    "You decide whether an item deserves interrupting the user. " "Reply with JSON only. Be strict."
)

_PROMPT = """\
Active threads:
{threads}

Items:
{items}

For each item return JSON:
{{"items": [{{"index": 0, "thread": "exact thread name or empty",
"interrupt": false, "movement": false, "reason": "short"}}]}}

Interrupt only when ALL of these hold:
- the item is a commitment, decision, open question, contradiction, or block
- it ties to exactly one active thread
- it is new movement, a contradiction, a block, or a commitment/decision with no owner

A topic mention is not enough. movement is true only when the item matches
that thread's "movement" definition.
"""


def _format_threads(threads: list[GoalThread]) -> str:
    if not threads:
        return "(none)"
    lines = []
    for thread in threads:
        lines.append(
            f"- {thread.name}\n"
            f"  why: {thread.why}\n"
            f"  movement: {thread.movement}\n"
            f"  interrupt: {thread.interrupt}"
        )
    return "\n".join(lines)


def _format_items(items: list[dict[str, Any]]) -> str:
    lines = []
    for index, item in enumerate(items):
        lines.append(
            f"{index}. [{item.get('kind')}] {item.get('text')} "
            f"(owner={item.get('owner') or '-'})"
        )
    return "\n".join(lines) or "(none)"


async def score_items(
    cfg: AgentConfig,
    items: list[dict[str, Any]],
    threads: list[GoalThread],
) -> list[dict[str, Any]]:
    """Attach thread, interrupt, movement, and reason onto copies of *items*."""
    scored = [dict(item) for item in items]
    for item in scored:
        item.setdefault("thread", "")
        item.setdefault("interrupt", False)
        item.setdefault("movement", False)
        item.setdefault("reason", "")
    if not scored or not threads:
        return scored
    prompt = _PROMPT.format(threads=_format_threads(threads), items=_format_items(scored))
    try:
        raw = await llm_run(cfg, prompt, system_prompt=_SYSTEM)
    except Exception as exc:
        logger.error("coworker score failed: %s", exc, exc_info=True)
        raise
    by_name = {thread.name.lower(): thread.name for thread in threads}
    by_slug = {thread.slug: thread.name for thread in threads}
    for entry in parse_items_payload(raw):
        raw_index = entry.get("index")
        if raw_index is None:
            continue
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if index < 0 or index >= len(scored):
            continue
        raw_thread = str(entry.get("thread") or "").strip()
        canonical = by_name.get(raw_thread.lower()) or by_slug.get(raw_thread.lower()) or ""
        interrupt = bool(entry.get("interrupt")) and bool(canonical)
        scored[index]["thread"] = canonical
        scored[index]["interrupt"] = interrupt
        scored[index]["movement"] = bool(entry.get("movement")) and bool(canonical)
        scored[index]["reason"] = str(entry.get("reason") or "").strip()
    return scored
