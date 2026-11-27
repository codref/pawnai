"""One-shot completions on the background model (``agent.default``).

Used by small side tasks (portal chat auto-naming, image captions for the
file index) that should not run a full agent turn. The model selection is
applied to a shallow copy of the config so the process-wide selection is
left alone.
"""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any, Optional

_JSON_RE = re.compile(r"\{.*\}", re.S)


def background_selection(cfg: Any, *, vision: bool = False) -> Optional[Any]:
    """Return a ``ModelSelection`` for the background model, or None.

    With *vision*, returns None unless the background model is flagged
    ``vision: true``.
    """
    from pawn_agent.utils.model_catalog import (  # noqa: PLC0415
        active_selection,
        apply_model_selection,
        get_background_model,
        model_is_vision,
    )

    if not hasattr(cfg, "agent"):
        return None
    background = get_background_model(cfg)
    if vision and not model_is_vision(cfg, background):
        return None
    local = copy.copy(cfg)
    if background:
        return apply_model_selection(local, background)
    return active_selection(local)


def complete_text(cfg: Any, messages: list[dict], *, session: str, vision: bool = False) -> str:
    """Run one completion on the background model; returns '' when unavailable."""
    from sallm.llm import complete  # noqa: PLC0415

    from pawn_agent.utils.model_catalog import completion_headers  # noqa: PLC0415

    selection = background_selection(cfg, vision=vision)
    if selection is None:
        return ""
    extra: dict[str, Any] = {}
    if selection.api_key:
        extra["api_key"] = selection.api_key
    headers = completion_headers(selection, session)
    if headers:
        extra["extra_headers"] = headers
    result = complete(
        model=selection.litellm_model,
        messages=messages,
        api_base=selection.api_base,
        **extra,
    )
    return str(result.get("content") or "")


def describe_image(cfg: Any, path: Path, prompt: str, *, session: str) -> str:
    """Caption one image file with the background vision model ('' if none)."""
    from sallm.attachments import image_part  # noqa: PLC0415

    parts: list[dict] = [{"type": "text", "text": prompt}, image_part(path)]
    return complete_text(cfg, [{"role": "user", "content": parts}], session=session, vision=True)


def parse_json_object(raw: str) -> dict[str, Any]:
    """First ``{...}`` in *raw* as a dict (tolerates fences and chatter)."""
    match = _JSON_RE.search(raw or "")
    if not match:
        return {}
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
