"""Vision pass for planner event screenshots (title / people / times).

Used when an Outlook clip includes images but DOM scrape / text left the
event thin (stub tab title, no attendees, …). Calls the background model
only when it is flagged ``vision: true``.
"""

from __future__ import annotations

import json
import logging
import re
import tempfile
from pathlib import Path
from typing import Any, Optional

from pawn_agent.core.vision import MAX_IMAGE_BYTES

logger = logging.getLogger(__name__)

_EXTRACT_PROMPT = (
    "Extract calendar / meeting fields from this screenshot of an Outlook "
    "or Teams event. Reply with ONLY a JSON object (no markdown fences):\n"
    '{"title":"","start":"","end":"","organizer":"","attendees":[],'
    '"location":"","meeting_url":"","topics":[]}\n'
    "title MUST be the meeting subject (e.g. from the large heading or "
    "'… - Meeting - Calendar' breadcrumb). NEVER use chrome labels: "
    "Tracking, Accepted, Chat, Join, Calendar, Outlook, Mail, Busy, "
    "Reminder, Prepare for this meeting, Microsoft Teams meeting.\n"
    "attendees are people display names only — never UI buttons "
    "(Chat, Join, Copilot) or '[image]'.\n"
    "Use local times as YYYY-MM-DDTHH:mm when possible. Leave unknowns empty."
)

_JSON_RE = re.compile(r"\{.*\}", re.S)


def needs_event_vision(
    *,
    title: str,
    start: str = "",
    attendees: list[str] | None = None,
    organizer: str = "",
) -> bool:
    """True when a screenshot pass would likely improve metadata."""
    from pawn_server.core.planner import is_stub_event_title  # noqa: PLC0415

    if is_stub_event_title(title):
        return True
    if not (start or "").strip():
        return True
    if not (attendees or []):
        return True
    if not (organizer or "").strip():
        return True
    return False


def extract_event_fields_from_images(
    cfg: Any,
    images: list[bytes],
) -> dict[str, Any]:
    """Run vision over one or more screenshots. Empty dict when skipped."""
    empty: dict[str, Any] = {
        "title": "",
        "start": "",
        "end": "",
        "organizer": "",
        "attendees": [],
        "location": "",
        "meeting_url": "",
        "topics": [],
    }
    usable = [b for b in images if b and len(b) <= MAX_IMAGE_BYTES]
    if not usable:
        return empty

    try:
        from pawn_agent.utils.config import load_config  # noqa: PLC0415
        from pawn_agent.utils.model_catalog import (  # noqa: PLC0415
            apply_model_selection,
            get_background_model,
            model_is_vision,
        )

        agent_cfg = cfg if cfg is not None and hasattr(cfg, "agent") else load_config()
        background = get_background_model(agent_cfg)
        if not model_is_vision(agent_cfg, background):
            logger.info(
                "event vision skipped: background model %r is not vision-enabled",
                background,
            )
            return empty
        if background:
            apply_model_selection(agent_cfg, background)

        # One model call with the last (usually whole-view) shot.
        raw = _complete_vision(agent_cfg, usable[-1])
    except Exception as exc:
        logger.warning("event vision call failed (non-fatal): %s", exc)
        return empty
    parsed = _parse_fields(raw)
    if parsed.get("title") or parsed.get("attendees"):
        logger.info(
            "event vision extracted title=%r attendees=%s",
            parsed.get("title"),
            parsed.get("attendees"),
        )
    return parsed


def _complete_vision(cfg: Any, data: bytes) -> str:
    from sallm.attachments import image_part  # noqa: PLC0415
    from sallm.llm import complete  # noqa: PLC0415

    from pawn_agent.utils.model_catalog import completion_headers  # noqa: PLC0415

    path = _write_temp(data)
    try:
        parts: list[dict] = [
            {"type": "text", "text": _EXTRACT_PROMPT},
            image_part(path),
        ]
        selection = cfg.model_selection
        extra: dict = {}
        if selection.api_key:
            extra["api_key"] = selection.api_key
        headers = completion_headers(selection, "pawn-event-vision")
        if headers:
            extra["extra_headers"] = headers
        result = complete(
            model=selection.litellm_model,
            messages=[{"role": "user", "content": parts}],
            api_base=selection.api_base,
            **extra,
        )
    finally:
        path.unlink(missing_ok=True)
    return str(result.get("content") or "")


def _write_temp(data: bytes) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="pawn-event-", suffix=".png", delete=False)
    try:
        handle.write(data)
    finally:
        handle.close()
    return Path(handle.name)


def _parse_fields(raw: str) -> dict[str, Any]:
    out: dict[str, Any] = {
        "title": "",
        "start": "",
        "end": "",
        "organizer": "",
        "attendees": [],
        "location": "",
        "meeting_url": "",
        "topics": [],
    }
    text = (raw or "").strip()
    if not text:
        return out
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = _JSON_RE.search(text)
    if not match:
        return out
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError:
        return out
    if not isinstance(data, dict):
        return out
    out["title"] = str(data.get("title") or "").strip()[:120]
    out["start"] = str(data.get("start") or "").strip()[:40]
    out["end"] = str(data.get("end") or "").strip()[:40]
    out["organizer"] = str(data.get("organizer") or "").strip()[:80]
    out["location"] = str(data.get("location") or "").strip()[:160]
    out["meeting_url"] = str(data.get("meeting_url") or "").strip()[:300]
    attendees = data.get("attendees") or []
    if isinstance(attendees, str):
        attendees = [attendees]
    out["attendees"] = [
        str(a).strip() for a in attendees if str(a).strip()
    ][:40]
    topics = data.get("topics") or []
    if isinstance(topics, str):
        topics = [topics]
    out["topics"] = [str(t).strip() for t in topics if str(t).strip()][:12]
    return out


def decode_snippet_images(snippets: list[dict[str, Any]]) -> list[bytes]:
    """Best-effort decode of image snippets for the vision pass."""
    from pawn_server.core.captures import CaptureError, _decode_image  # noqa: PLC0415

    out: list[bytes] = []
    for raw in snippets:
        if str(raw.get("kind") or "text").strip().lower() != "image":
            continue
        try:
            out.append(_decode_image(str(raw.get("data_base64") or "")))
        except CaptureError:
            continue
        except Exception:  # noqa: BLE001
            continue
    return out
