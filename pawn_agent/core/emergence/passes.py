"""Model passes over one window: triage and typed extraction."""

from __future__ import annotations

import json
import logging
from typing import Any

from pawn_agent.core.coworker.jsonutil import parse_json_value
from pawn_agent.core.emergence import prompts
from pawn_agent.core.llm_sub import run as llm_run

logger = logging.getLogger(__name__)

_TEXT_MAX = 300
_CONTEXT_MAX = 500
_QUOTE_MAX = 1000


class PassError(RuntimeError):
    """A model call failed; carries the original exception."""


def _clip(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split()) if limit <= _TEXT_MAX else str(value or "")
    text = text.strip()
    if len(text) <= limit:
        return text
    cut = text[:limit].rsplit(" ", 1)[0].rstrip()
    return cut or text[:limit]


def _score(value: Any) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError):
        return 0.0
    if number > 1.0 and number <= 100.0:
        number = number / 100.0
    return max(0.0, min(1.0, number))


def parse_triage(raw: str, types: list[str]) -> dict[str, float] | None:
    """Scores per type, or None when the reply is unusable."""
    try:
        data = parse_json_value(raw)
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(data, dict) and isinstance(data.get("scores"), dict):
        data = data["scores"]
    if not isinstance(data, dict):
        return None
    lowered = {str(k).strip().lower(): v for k, v in data.items()}
    if not any(t in lowered for t in types):
        return None
    return {t: _score(lowered.get(t)) for t in types}


async def triage(
    cfg: Any, types: list[str], source: str, part: str
) -> tuple[dict[str, float], bool]:
    """Return ``(scores, ok)``. Fails open: every type scores 1.0 on failure.

    Raises :class:`PassError` when the model call itself fails.
    """
    try:
        raw = await llm_run(
            cfg,
            prompts.triage_prompt(types, source, part),
            system_prompt=prompts.SYSTEM,
            think=False,
        )
    except Exception as exc:
        raise PassError(str(exc)) from exc
    scores = parse_triage(raw, types)
    if scores is None:
        logger.warning("emergence triage reply unusable; extracting every type")
        return {t: 1.0 for t in types}, False
    return scores, True


def _list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_clip(v, _TEXT_MAX) for v in value if str(v or "").strip()]
    text = str(value or "").strip()
    return [text] if text else []


def normalize_observation(entry: dict[str, Any], type_name: str) -> dict[str, Any] | None:
    """Shape one raw model observation. None when it has no claim."""
    text = _clip(entry.get("text"), _TEXT_MAX)
    if not text:
        return None
    spec = prompts.SPECS[type_name]
    fields: dict[str, Any] = {}
    for key in spec.fields:
        value = entry.get(key)
        if key in {"assumptions", "alternatives"}:
            fields[key] = _list(value)
        elif key == "blocking":
            fields[key] = bool(value) and str(value).strip().lower() not in {"false", "no", "0"}
        else:
            fields[key] = _clip(value, _TEXT_MAX)
    return {
        "type": type_name,
        "text": text,
        "context": _clip(entry.get("context"), _CONTEXT_MAX),
        "speaker_name": _clip(entry.get("speaker"), 120),
        "quote": _clip(entry.get("quote"), _QUOTE_MAX),
        "confidence": _score(entry.get("confidence")) if "confidence" in entry else None,
        "fields": fields,
        "extractor": prompts.extractor_tag(type_name),
    }


_RETRY_NOTE = (
    "\n\nYour previous reply to this request was not valid JSON. Reply again "
    "with only the JSON object, double-quoted keys, no trailing commas."
)


def parse_observations(raw: str, type_name: str) -> list[dict[str, Any]] | None:
    """Observations from a reply; ``None`` when the reply is not JSON."""
    try:
        data = parse_json_value(raw)
    except (json.JSONDecodeError, ValueError) as exc:
        logger.warning("emergence %s reply unparseable: %s", type_name, exc)
        return None
    if isinstance(data, dict):
        data = data.get("observations") or data.get("items") or []
    if not isinstance(data, list):
        return []
    out: list[dict[str, Any]] = []
    for entry in data:
        if isinstance(entry, dict):
            normalized = normalize_observation(entry, type_name)
            if normalized is not None:
                out.append(normalized)
    return out


async def extract_type(
    cfg: Any,
    type_name: str,
    source: str,
    part: str,
    *,
    think: bool = False,
    examples: dict[str, list[dict[str, str]]] | None = None,
    spec: Any = None,
) -> list[dict[str, Any]]:
    """Run one type's prompt over a window. Raises :class:`PassError`.

    A reply that is not JSON is retried once; a second bad reply yields an
    empty list.
    """
    prompt = prompts.typed_prompt(type_name, source, part, examples, spec=spec)
    for attempt in range(2):
        try:
            raw = await llm_run(
                cfg,
                prompt if attempt == 0 else prompt + _RETRY_NOTE,
                system_prompt=prompts.SYSTEM,
                think=think,
            )
        except Exception as exc:
            raise PassError(str(exc)) from exc
        found = parse_observations(raw, type_name)
        if found is not None:
            return found
        logger.warning("emergence %s: retrying after unparseable reply", type_name)
    return []
