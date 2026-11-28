"""Extract structured items from a transcript or note."""

from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from typing import Any, Optional

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

# Window defaults when ``coworker.extract_window_*`` is unset (~3k tokens).
_WINDOW_CHARS = 12000
_WINDOW_OVERLAP = 1500

# Header written by ``pipeline._source_with_annotations``.
ANNOTATIONS_HEADING = "## User annotations"
TRANSCRIPT_HEADING = "## Transcript"

# ``[mm:ss.ss] Speaker:`` turn lines from ``render_analysis_transcript``.
_TURN_RE = re.compile(r"^\[(\d+):(\d+(?:\.\d+)?)\]\s")
_WORD_RE = re.compile(r"[\w']+", re.UNICODE)

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
{part}
SOURCE:
{source}
"""

_PART = """
This SOURCE is part {index} of {total} of a longer source, with a small
overlap with the neighbouring parts. Extract only what this part says.
Do not mention parts in any field.
{annotations}"""


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


@dataclass(frozen=True)
class Window:
    """One slice of the source sent to the model."""

    index: int
    text: str
    char_start: int


def split_annotations(source_text: str) -> tuple[str, str]:
    """Split ``(annotations, body)`` from the pipeline's annotated source."""
    text = source_text or ""
    if not text.startswith(ANNOTATIONS_HEADING):
        return "", text
    marker = f"\n\n{TRANSCRIPT_HEADING}\n"
    cut = text.find(marker)
    if cut < 0:
        return "", text
    annotations = text[len(ANNOTATIONS_HEADING) : cut].strip()
    return annotations, text[cut + len(marker) :]


def _units(text: str) -> list[tuple[int, str]]:
    """Split *text* into ``(offset, unit)`` speaker turns or paragraphs."""
    lines = text.splitlines(keepends=True)
    is_transcript = any(_TURN_RE.match(line) for line in lines[:50])
    units: list[tuple[int, str]] = []
    start = 0
    buf: list[str] = []
    offset = 0
    for line in lines:
        boundary = _TURN_RE.match(line) if is_transcript else not line.strip()
        if boundary and buf:
            units.append((start, "".join(buf)))
            buf = []
        if not buf:
            start = offset
        buf.append(line)
        offset += len(line)
    if buf:
        units.append((start, "".join(buf)))
    return units


def _hard_split(offset: int, unit: str, max_chars: int) -> list[tuple[int, str]]:
    """Split one oversized unit on line, then character, boundaries."""
    out: list[tuple[int, str]] = []
    start = offset
    buf = ""
    for line in unit.splitlines(keepends=True):
        while len(line) > max_chars:
            if buf:
                out.append((start, buf))
                start += len(buf)
                buf = ""
            out.append((start, line[:max_chars]))
            start += max_chars
            line = line[max_chars:]
        if len(buf) + len(line) > max_chars and buf:
            out.append((start, buf))
            start += len(buf)
            buf = ""
        buf += line
    if buf:
        out.append((start, buf))
    return out


def window_source(
    text: str,
    *,
    max_chars: int = _WINDOW_CHARS,
    overlap_chars: int = _WINDOW_OVERLAP,
) -> list[Window]:
    """Pack speaker turns (or paragraphs) into overlapping windows.

    Each window holds whole units up to *max_chars*. The next window starts
    with the trailing units of the previous one, up to *overlap_chars*, so a
    decision that spans a cut is seen whole at least once.
    """
    body = text or ""
    if not body.strip():
        return []
    max_chars = max(500, int(max_chars))
    overlap_chars = max(0, min(int(overlap_chars), max_chars // 2))
    if len(body) <= max_chars:
        return [Window(index=0, text=body, char_start=0)]
    units: list[tuple[int, str]] = []
    for offset, unit in _units(body):
        if len(unit) > max_chars:
            units.extend(_hard_split(offset, unit, max_chars))
        else:
            units.append((offset, unit))

    windows: list[Window] = []
    i = 0
    while i < len(units):
        size = 0
        j = i
        while j < len(units) and (j == i or size + len(units[j][1]) <= max_chars):
            size += len(units[j][1])
            j += 1
        chunk = "".join(unit for _, unit in units[i:j])
        windows.append(Window(index=len(windows), text=chunk, char_start=units[i][0]))
        if j >= len(units):
            break
        # Step back over trailing units that fit in the overlap budget.
        back = j
        carried = 0
        while back - 1 > i and carried + len(units[back - 1][1]) <= overlap_chars:
            back -= 1
            carried += len(units[back][1])
        i = back
    return windows


def _words(text: str) -> list[tuple[str, int]]:
    return [(m.group(0).lower(), m.start()) for m in _WORD_RE.finditer(text or "")]


def locate_quote(window_text: str, quote: str) -> tuple[Optional[int], float]:
    """Return ``(char offset in window, match score 0..1)`` for *quote*.

    Matches on lowercased words, so speaker-line formatting differences
    between the quote and the transcript do not matter.
    """
    quote_words = [w for w, _ in _words(quote)]
    window_words = _words(window_text)
    if not quote_words or not window_words:
        return None, 0.0
    matcher = difflib.SequenceMatcher(
        None, [w for w, _ in window_words], quote_words, autojunk=False
    )
    blocks = [b for b in matcher.get_matching_blocks() if b.size]
    if not blocks:
        return None, 0.0
    matched = sum(b.size for b in blocks)
    score = matched / len(quote_words)
    # Anchor on the first block that is not a stray common word.
    anchor = next((b for b in blocks if b.size >= 3), max(blocks, key=lambda b: b.size))
    return window_words[anchor.a][1], score


def start_seconds_at(text: str, offset: int) -> Optional[float]:
    """Seconds of the last ``[mm:ss.ss]`` turn at or before *offset*."""
    best: Optional[float] = None
    pos = 0
    for line in (text or "").splitlines(keepends=True):
        if pos > offset:
            break
        match = _TURN_RE.match(line)
        if match:
            best = int(match.group(1)) * 60 + float(match.group(2))
        pos += len(line)
    return best


_SPEAKER_RE = re.compile(r"^\[\d+:\d+(?:\.\d+)?\]\s+(.+?):\s*$")


def speaker_at(text: str, offset: int) -> Optional[str]:
    """Speaker label of the last ``[mm:ss.ss] Name:`` turn at or before *offset*."""
    best: Optional[str] = None
    pos = 0
    for line in (text or "").splitlines(keepends=True):
        if pos > offset:
            break
        match = _SPEAKER_RE.match(line.rstrip("\n"))
        if match:
            best = match.group(1).strip()
        pos += len(line)
    return best


def _norm(text: str) -> str:
    return " ".join(w for w, _ in _words(text))


def _similar(a: str, b: str, threshold: float) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na or not nb:
        return False
    if na == nb:
        return True
    return difflib.SequenceMatcher(None, na, nb, autojunk=False).ratio() >= threshold


def claims_similar(a: str, b: str, threshold: float = 0.85) -> bool:
    """Whether two claims say nearly the same thing (word-normalized)."""
    return _similar(a, b, threshold)


def _richness(item: dict[str, Any]) -> tuple[float, int]:
    return (
        float(item.get("quote_score") or 0.0),
        len(item.get("context") or "") + len(item.get("quote") or ""),
    )


def merge_items(items: list[dict[str, Any]], *, kind_key: str = "kind") -> list[dict[str, Any]]:
    """Drop duplicates from overlapping windows, keeping the richer copy.

    Two items are the same when they share a kind (``kind_key``) and either
    their claims are near-identical or their quotes are.
    """
    kept: list[dict[str, Any]] = []
    for item in items:
        dup_index = None
        for index, other in enumerate(kept):
            if other.get(kind_key) != item.get(kind_key):
                continue
            if _similar(other.get("text") or "", item.get("text") or "", 0.85) or (
                (item.get("quote") or "").strip()
                and _similar(other.get("quote") or "", item.get("quote") or "", 0.8)
            ):
                dup_index = index
                break
        if dup_index is None:
            kept.append(item)
        elif _richness(item) > _richness(kept[dup_index]):
            kept[dup_index] = item
    kept.sort(
        key=lambda it: (
            it.get("start_sec") is None,
            it.get("start_sec") or 0.0,
            it.get("window", 0),
        )
    )
    return kept


def _window_settings(cfg: AgentConfig) -> tuple[int, int]:
    coworker = getattr(cfg, "coworker", None)
    size = int(getattr(coworker, "extract_window_chars", 0) or _WINDOW_CHARS)
    overlap = int(getattr(coworker, "extract_window_overlap", _WINDOW_OVERLAP) or 0)
    return size, overlap


def _build_prompt(window: Window, total: int, annotations: str) -> str:
    if total <= 1:
        source = window.text
        if annotations:
            source = f"{ANNOTATIONS_HEADING}\n{annotations}\n\n{TRANSCRIPT_HEADING}\n{window.text}"
        return _PROMPT.format(part="", source=source)
    note = ""
    if annotations:
        note = f"\nUser annotations about the whole source (context, not items):\n{annotations}\n"
    part = _PART.format(index=window.index + 1, total=total, annotations=note)
    return _PROMPT.format(part=part, source=window.text)


async def extract_items(cfg: AgentConfig, source_text: str) -> list[dict[str, Any]]:
    """Return normalized item dicts across the whole source.

    Long sources are split into overlapping windows (one model call each),
    merged, and deduplicated. Items whose quote is found in the source carry
    ``start_sec`` (transcripts) and ``quote_score``. A window whose call fails
    is skipped; the error is raised only when every window fails.
    """
    if not (source_text or "").strip():
        return []
    annotations, body = split_annotations(source_text)
    size, overlap = _window_settings(cfg)
    windows = window_source(body, max_chars=size, overlap_chars=overlap)
    collected: list[dict[str, Any]] = []
    failures = 0
    last_exc: Optional[Exception] = None
    for window in windows:
        try:
            raw = await llm_run(
                cfg,
                _build_prompt(window, len(windows), annotations),
                system_prompt=_SYSTEM,
            )
        except Exception as exc:
            failures += 1
            last_exc = exc
            logger.error(
                "coworker extract failed on window %d/%d: %s",
                window.index + 1,
                len(windows),
                exc,
                exc_info=True,
            )
            continue
        for entry in parse_items_payload(raw):
            normalized = normalize_extracted_entry(entry)
            if normalized is None:
                continue
            offset, score = locate_quote(window.text, normalized["quote"])
            normalized["quote_score"] = round(score, 3)
            normalized["window"] = window.index
            normalized["start_sec"] = (
                start_seconds_at(body, window.char_start + offset) if offset is not None else None
            )
            collected.append(normalized)
    if windows and failures == len(windows) and last_exc is not None:
        raise last_exc
    if len(windows) > 1:
        logger.info(
            "coworker extract: %d windows, %d raw items, %d failed",
            len(windows),
            len(collected),
            failures,
        )
    return merge_items(collected)
