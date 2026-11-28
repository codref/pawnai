"""Side-by-side report: single-prompt items vs. observation-layer items.

Used by ``pawn-server coworker compare`` to check Phase 1's exit test on
real meetings. Nothing is stored; the user ticks useful items in the note.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from pawn_agent.core.coworker.extract import (
    claims_similar,
    extract_items,
    split_annotations,
    window_source,
)
from pawn_agent.core.coworker.notes import format_offset
from pawn_agent.core.emergence.pipeline import ObserveResult, observe_source, project_items
from pawn_agent.core.emergence.types import ITEM_KINDS


@dataclass
class Comparison:
    source_ref: str
    baseline: list[dict[str, Any]]
    baseline_calls: int
    baseline_seconds: float
    observed: ObserveResult
    items: list[dict[str, Any]]


async def run_comparison(cfg: Any, *, source_ref: str, source_text: str) -> Comparison:
    """Run both extractors on the same source text (no DB writes)."""
    _, body = split_annotations(source_text)
    coworker = getattr(cfg, "coworker", None)
    windows = window_source(
        body,
        max_chars=int(getattr(coworker, "extract_window_chars", 12000) or 12000),
        overlap_chars=int(getattr(coworker, "extract_window_overlap", 1500) or 0),
    )
    started = time.monotonic()
    baseline = await extract_items(cfg, source_text)
    baseline_seconds = round(time.monotonic() - started, 1)
    observed = await observe_source(
        cfg,
        source_kind="session",
        source_ref=source_ref,
        source_text=source_text,
        persist=False,
    )
    return Comparison(
        source_ref=source_ref,
        baseline=baseline,
        baseline_calls=len(windows),
        baseline_seconds=baseline_seconds,
        observed=observed,
        items=project_items(observed.observations),
    )


def _matches(item: dict[str, Any], others: list[dict[str, Any]]) -> bool:
    return any(
        claims_similar(item.get("text") or "", o.get("text") or "", 0.6)
        or (
            (item.get("quote") or "").strip()
            and claims_similar(item.get("quote") or "", o.get("quote") or "", 0.6)
        )
        for o in others
    )


def _at(seconds: Any) -> str:
    text = format_offset(seconds)
    return f"[{text}] " if text else ""


def _item_lines(item: dict[str, Any], *, also: bool) -> list[str]:
    owner = f" — owner: {item['owner']}" if item.get("owner") else ""
    due = f", due {item['due']}" if item.get("due") else ""
    mark = " · *in both*" if also else ""
    if item.get("rank_score") is not None:
        mark += f" · rank {item['rank_score']:.2f}"
    lines = [
        f"- [ ] {_at(item.get('start_sec'))}**{item.get('kind')}** {item.get('text')}"
        f"{owner}{due}{mark}"
    ]
    quote = " ".join((item.get("quote") or "").split())
    if quote:
        lines.append(f"    > {quote[:300]}")
    return lines


def render_report(cmp: Comparison) -> str:
    """Markdown note with counts, both item lists, and the other observations."""
    stats = cmp.observed.stats
    new_only = [i for i in cmp.items if not _matches(i, cmp.baseline)]
    old_only = [i for i in cmp.baseline if not _matches(i, cmp.items)]
    others: dict[str, list[dict[str, Any]]] = {}
    for obs in cmp.observed.live:
        if obs["type"] not in ITEM_KINDS:
            others.setdefault(obs["type"], []).append(obs)
    calls = int(stats.get("triage_calls", 0)) + int(stats.get("typed_calls", 0))
    out = [
        "---",
        "pawn: extract-compare",
        f"source_ref: {cmp.source_ref}",
        f"model: {stats.get('model') or ''}",
        "reviewed: false",
        "---",
        "",
        f"# Extraction compare: {cmp.source_ref}",
        "",
        "Tick `[x]` on every item you would have wanted. Precision is ticked / listed",
        "per column; items only one side found show recall differences.",
        "When done, set `reviewed: true` above: unticked lines then count as",
        "*not wanted* when Pawn learns from this note (`coworker labels import`).",
        "",
        "| | Single prompt | Observations |",
        "|---|---|---|",
        f"| Items | {len(cmp.baseline)} | {len(cmp.items)} |",
        f"| Found only here | {len(old_only)} | {len(new_only)} |",
        f"| Model calls | {cmp.baseline_calls} | {calls} |",
        f"| Seconds | {cmp.baseline_seconds} | {stats.get('seconds', 0)} |",
        f"| Windows triage scored quiet | — | {stats.get('windows_skipped', 0)} of "
        f"{stats.get('windows', 0)} |",
        f"| Rejected (quote not in source) | — | {stats.get('rejected', 0)} |",
        f"| Other observations (not items) | — | " f"{sum(len(v) for v in others.values())} |",
        "",
        f"## Observation items ({len(cmp.items)})",
        "",
    ]
    for item in cmp.items:
        out.extend(_item_lines(item, also=item not in new_only))
    out += ["", f"## Single-prompt items ({len(cmp.baseline)})", ""]
    for item in cmp.baseline:
        out.extend(_item_lines(item, also=item not in old_only))
    if others:
        out += ["", "## Other observations (stored, not items)", ""]
        for type_name, rows in others.items():
            out.append(f"### {type_name} ({len(rows)})")
            out.append("")
            for obs in rows:
                speaker = f" — {obs['speaker_name']}" if obs.get("speaker_name") else ""
                if obs.get("rank_score") is not None:
                    speaker += f" · rank {obs['rank_score']:.2f}"
                out.append(f"- [ ] {_at(obs.get('start_sec'))}{obs['text']}{speaker}")
            out.append("")
    rejected = cmp.observed.rejected
    if rejected:
        out += ["", f"## Rejected ({len(rejected)})", ""]
        out.append("Tick any that were real (the quote check was wrong).")
        out.append("")
        for obs in rejected:
            out.append(
                f"- [ ] **{obs['type']}** {obs['text']} "
                f"({obs.get('reject_reason')}, match {obs.get('quote_score')})"
            )
    return "\n".join(out).rstrip() + "\n"
