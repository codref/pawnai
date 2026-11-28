"""JSON helpers for structured coworker LLM replies."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
_TRAILING_COMMA_RE = re.compile(r",(\s*[}\]])")


def _loads_lenient(text: str) -> Any:
    """``json.loads`` that tolerates what local models commonly emit.

    Raw control characters inside strings, trailing commas before ``}`` or
    ``]``, and text after the first complete value.
    """
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
    fixed = _TRAILING_COMMA_RE.sub(r"\1", text)
    try:
        return json.loads(fixed, strict=False)
    except json.JSONDecodeError:
        value, _ = json.JSONDecoder(strict=False).raw_decode(fixed)
        return value


_KINDS = frozenset(
    {"decision", "commitment", "open_question", "block", "contradiction", "proposal"}
)


def parse_json_value(raw: str) -> Any:
    """Parse a JSON object or list, tolerating a markdown fence."""
    text = (raw or "").strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1).strip()
    try:
        return _loads_lenient(text)
    except json.JSONDecodeError:
        start_obj = text.find("{")
        start_arr = text.find("[")
        starts = [i for i in (start_obj, start_arr) if i >= 0]
        if not starts:
            raise
        start = min(starts)
        end = max(text.rfind("}"), text.rfind("]"))
        if end <= start:
            raise
        try:
            return _loads_lenient(text[start : end + 1])
        except json.JSONDecodeError:
            return _loads_lenient(text[start:])


def parse_items_payload(raw: str) -> list[dict[str, Any]]:
    """Return extracted item dicts, or an empty list when the model misbehaves."""
    try:
        data = parse_json_value(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("coworker JSON parse failed: %s", exc)
        return []
    if isinstance(data, dict):
        data = data.get("items") or data.get("results") or []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in data:
        if isinstance(entry, dict):
            out.append(entry)
    return out


def normalize_kind(value: Any) -> str:
    kind = str(value or "").strip().lower().replace(" ", "_").replace("-", "_")
    if kind in {"question", "openquestion"}:
        kind = "open_question"
    if kind not in _KINDS:
        return "open_question"
    return kind
