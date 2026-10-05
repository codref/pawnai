"""Weekly review note, including a proposed Goals.md the user applies by hand."""

from __future__ import annotations

from collections import Counter
from typing import Any

from pawn_core.goals import Goals
from pawn_core.vault import dump_frontmatter


def render_review(
    *,
    week: str,
    goals: Goals,
    items: list[dict[str, Any]],
    stale: list[dict[str, Any]],
    commitments: list[dict[str, Any]],
    ownerless: list[dict[str, Any]],
    themes: list[str],
) -> str:
    """Render ``Pawn/Reviews/{week}.md``."""
    lines = [f"# Review {week}", ""]
    lines.append("## Threads")
    if not goals.active:
        lines.append("")
        lines.append("_No active threads._")
    for thread in goals.active:
        related = [item for item in items if (item.get("thread") or "") == thread.name]
        moved = [item for item in related if item.get("movement")]
        lines.append("")
        lines.append(f"### {thread.name}")
        lines.append(f"- open: {len(related)}")
        lines.append(f"- moved: {len(moved)}")
        for item in related[:8]:
            lines.append(f"- {item.get('kind')}: {item.get('text')}")

    lines.append("")
    lines.append("## Forgotten commitments")
    if not commitments:
        lines.append("")
        lines.append("_None._")
    for item in commitments:
        lines.append(f"- {item.get('text')} ({item.get('short_id')})")

    lines.append("")
    lines.append("## Ownerless decisions")
    if not ownerless:
        lines.append("")
        lines.append("_None._")
    for item in ownerless:
        lines.append(f"- {item.get('text')} ({item.get('short_id')})")

    lines.append("")
    lines.append("## Stale threads")
    if not stale:
        lines.append("")
        lines.append("_None._")
    for stale_thread in stale:
        lines.append(f"- {stale_thread.get('name')}")

    lines.append("")
    lines.append("## Themes outside your goals")
    if not themes:
        lines.append("")
        lines.append("_None._")
    for theme in themes:
        lines.append(f"- {theme}")

    dismissed = [item for item in items if item.get("status") == "dismissed"]
    lines.append("")
    lines.append("## Ignore-list suggestions")
    counts = Counter((item.get("thread") or item.get("kind") or "") for item in dismissed)
    if not counts:
        lines.append("")
        lines.append("_None._")
    for name, count in counts.most_common(8):
        if name:
            lines.append(f"- {name} ({count} ignored)")

    acted = sum(1 for item in items if item.get("status") in {"filed", "task"})
    ignored = len(dismissed)
    snoozed = sum(1 for item in items if item.get("status") == "snoozed")
    notified = sum(1 for item in items if item.get("interrupt"))
    lines.append("")
    lines.append("## Feedback")
    lines.append(f"- pings: {notified}")
    lines.append(f"- acted: {acted}")
    lines.append(f"- ignored: {ignored}")
    lines.append(f"- snoozed: {snoozed}")

    lines.append("")
    lines.append("## Proposed goals")
    lines.append("")
    lines.append("```goals")
    lines.append(_proposed_goals(goals, stale))
    lines.append("```")
    lines.append("")
    return dump_frontmatter({"pawn": "review", "week": week}, "\n".join(lines))


def _proposed_goals(goals: Goals, stale: list[dict[str, Any]]) -> str:
    stale_names = {(thread.get("name") or "").lower() for thread in stale}
    lines = ["---", "pawn: goals"]
    if goals.timezone:
        lines.append(f"timezone: {goals.timezone}")
    lines.append("notify:")
    lines.append(f"  max_per_day: {goals.max_per_day}")
    if goals.quiet_hours:
        lines.append(f'  quiet_hours: "{goals.quiet_hours}"')
    if goals.ignore:
        lines.append("ignore:")
        for entry in goals.ignore:
            lines.append(f"  - {entry}")
    lines.append("---")
    lines.append("")
    lines.append("## Active")
    for thread in goals.active:
        if thread.name.lower() in stale_names:
            continue
        lines.extend(_thread_block(thread))
    lines.append("")
    lines.append("## Parked")
    for thread in goals.parked:
        lines.extend(_thread_block(thread))
    for thread in goals.active:
        if thread.name.lower() in stale_names:
            lines.extend(_thread_block(thread))
    lines.append("")
    return "\n".join(lines)


def _thread_block(thread: Any) -> list[str]:
    lines = ["", f"### {thread.name}"]
    if thread.why:
        lines.append(f"- why: {thread.why}")
    if thread.movement:
        lines.append(f"- movement: {thread.movement}")
    if thread.interrupt:
        lines.append(f"- interrupt: {thread.interrupt}")
    if thread.note:
        lines.append(f"- note: {thread.note}")
    return lines


def extract_goals_block(text: str) -> str:
    """Return the fenced ```goals body, or empty string."""
    marker = "```goals"
    start = text.find(marker)
    if start < 0:
        return ""
    body = text[start + len(marker) :]
    end = body.find("```")
    if end < 0:
        return ""
    return body[:end].strip() + "\n"
