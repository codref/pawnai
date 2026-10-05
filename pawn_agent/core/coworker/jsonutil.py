"""JSON helpers for structured coworker LLM replies."""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

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
        return json.loads(text)
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
        return json.loads(text[start : end + 1])


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
