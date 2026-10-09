"""Link a new item to related knowledge-index hits.

Retrieval follows sallm's asymmetric embedding pattern (instruct query,
raw documents) and adds quality gates: cosine-distance floor, proper-noun
anchors for transcript hits, source_ref dedupe, and preference for prior
items over raw meeting windows.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from pawn_agent.utils.config import AgentConfig

logger = logging.getLogger(__name__)

_TRANSCRIPT_KINDS = frozenset({"transcript", "session"})
_ITEM_KINDS = frozenset({"item"})

# Meeting filler and common English that capitalize at sentence start.
_ANCHOR_STOP = frozenset(
    {
        "the",
        "this",
        "that",
        "these",
        "those",
        "there",
        "their",
        "then",
        "than",
        "they",
        "them",
        "with",
        "from",
        "about",
        "after",
        "before",
        "because",
        "when",
        "what",
        "which",
        "where",
        "while",
        "would",
        "could",
        "should",
        "shall",
        "will",
        "have",
        "has",
        "had",
        "been",
        "being",
        "were",
        "was",
        "are",
        "is",
        "am",
        "and",
        "but",
        "for",
        "not",
        "you",
        "your",
        "our",
        "his",
        "her",
        "its",
        "also",
        "just",
        "like",
        "into",
        "onto",
        "over",
        "under",
        "again",
        "still",
        "only",
        "some",
        "any",
        "all",
        "each",
        "every",
        "other",
        "another",
        "more",
        "most",
        "such",
        "same",
        "team",
        "issue",
        "response",
        "question",
        "meeting",
        "today",
        "tomorrow",
        "yesterday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "january",
        "february",
        "march",
        "april",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "speaker",
        "okay",
        "yeah",
        "yes",
        "no",
    }
)

_CAP_WORD_RE = re.compile(r"\b([A-Z][a-zA-Z0-9][a-zA-Z0-9-]{1,})\b")


def extract_anchors(text: str, quote: str = "") -> list[str]:
    """Proper-noun-ish tokens from the summary and quote (order preserved)."""
    blob = f"{text or ''} {quote or ''}".strip()
    if not blob:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for match in _CAP_WORD_RE.finditer(blob):
        word = match.group(1)
        key = word.lower()
        if key in _ANCHOR_STOP or key in seen:
            continue
        seen.add(key)
        out.append(word)
    return out


def compose_related_query(text: str, quote: str = "") -> str:
    """Retrieval sentence: summary plus quote, without dumping the whole meeting."""
    summary = " ".join((text or "").split()).strip()
    snippet = " ".join((quote or "").split()).strip()
    if snippet and len(snippet) > 240:
        snippet = snippet[:240].rstrip() + "…"
    if summary and snippet:
        return f"{summary}\nQuote: {snippet}"
    return summary or snippet


def _hit_has_anchor(hit: dict[str, Any], anchors: list[str]) -> bool:
    blob = (hit.get("text") or "").lower()
    heading = (hit.get("heading") or "").lower()
    ref = (hit.get("source_ref") or "").lower()
    haystack = f"{blob}\n{heading}\n{ref}"
    return any(anchor.lower() in haystack for anchor in anchors)


def _transcript_ok(
    hit: dict[str, Any],
    *,
    anchors: list[str],
    max_distance: float,
) -> bool:
    """Transcript windows need an anchor match, or a tighter distance alone."""
    kind = (hit.get("source_kind") or "").strip().lower()
    if kind not in _TRANSCRIPT_KINDS:
        return True
    if anchors:
        return _hit_has_anchor(hit, anchors)
    distance = float(hit.get("distance") if hit.get("distance") is not None else 2.0)
    return distance <= max_distance * 0.75


def select_related_hits(
    hits: list[dict[str, Any]],
    *,
    source_ref: str,
    anchors: list[str],
    max_distance: float,
    limit: int,
) -> tuple[list[dict[str, Any]], int]:
    """Filter, dedupe, prefer items, and cap. Returns (display hits, recurrence)."""
    origin = (source_ref or "").strip()
    items: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    for hit in hits:
        ref = (hit.get("source_ref") or "").strip()
        if not ref or ref == origin:
            continue
        distance = float(hit.get("distance") if hit.get("distance") is not None else 2.0)
        if distance > max_distance:
            continue
        kind = (hit.get("source_kind") or "").strip().lower()
        if kind in _ITEM_KINDS:
            items.append(hit)
            continue
        if not _transcript_ok(hit, anchors=anchors, max_distance=max_distance):
            continue
        others.append(hit)

    def _best_by_ref(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        best: dict[str, dict[str, Any]] = {}
        for hit in rows:
            ref = hit["source_ref"]
            prev = best.get(ref)
            if prev is None or float(hit.get("distance") or 2.0) < float(
                prev.get("distance") or 2.0
            ):
                best[ref] = hit
        return sorted(best.values(), key=lambda h: float(h.get("distance") or 2.0))

    item_hits = _best_by_ref(items)
    other_hits = _best_by_ref(others)
    # Drop non-item sources that duplicate an item's note path.
    item_refs = {h["source_ref"] for h in item_hits}
    other_hits = [h for h in other_hits if h["source_ref"] not in item_refs]

    recurrence = len(item_hits)
    merged = item_hits + other_hits
    return merged[: max(0, int(limit))], recurrence


def format_related_label(hit: dict[str, Any]) -> str:
    ref = hit.get("source_ref") or ""
    heading = hit.get("heading") or ""
    label = f"[[{ref}]]" if ref.endswith(".md") or "/" in ref else ref
    if heading:
        label = f"{label} — {heading}"
    return label


def related_lines(
    cfg: AgentConfig,
    text: str,
    source_ref: str,
    *,
    quote: str = "",
    limit: int | None = None,
) -> tuple[list[str], int]:
    """Return wiki-link lines and a recurrence count of similar prior items."""
    from pawn_core.knowledge_index import search_chunks  # noqa: PLC0415

    coworker = cfg.coworker
    cap = int(limit if limit is not None else getattr(coworker, "related_limit", 5) or 5)
    max_distance = float(getattr(coworker, "related_max_distance", 0.45) or 0.45)
    fetch_mult = int(getattr(coworker, "related_fetch_multiplier", 6) or 6)

    query = compose_related_query(text, quote)
    if not query:
        return [], 0
    anchors = extract_anchors(text, quote)

    hits = search_chunks(
        cfg,
        query,
        limit=cap,
        max_distance=max_distance,
        fetch_multiplier=fetch_mult,
    )
    # Prefer prior items: also pull a dedicated item slice when the mixed
    # top-K was dominated by transcript mush.
    if cap > 0:
        item_hits = search_chunks(
            cfg,
            query,
            limit=cap,
            kind="item",
            max_distance=max_distance,
            fetch_multiplier=max(2, fetch_mult // 2),
        )
        seen_ids = {(h.get("source_ref"), h.get("text")) for h in hits}
        for hit in item_hits:
            key = (hit.get("source_ref"), hit.get("text"))
            if key not in seen_ids:
                hits.append(hit)
                seen_ids.add(key)

    selected, recurrence = select_related_hits(
        hits,
        source_ref=source_ref,
        anchors=anchors,
        max_distance=max_distance,
        limit=cap,
    )
    return [format_related_label(hit) for hit in selected], recurrence


def format_hits(hits: list[dict[str, Any]]) -> str:
    if not hits:
        return "(no matches)"
    rows = []
    for hit in hits:
        ref = hit.get("source_ref") or ""
        heading = hit.get("heading") or ""
        snippet = (hit.get("text") or "").replace("\n", " ")[:180]
        dist = hit.get("distance")
        dist_s = f" d={dist:.3f}" if isinstance(dist, (int, float)) else ""
        rows.append(f"- {hit.get('source_kind')} {ref} {heading}{dist_s}: {snippet}")
    return "\n".join(rows) + "\n"
