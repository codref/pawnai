"""Shared SiYuan Notes HTTP primitives.

These utilities are used by both pawn_diarize and pawn_agent.  Higher-level
helpers (SiyuanClient, document formatters) live in their respective packages.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Optional

import requests


def siyuan_post(
    base_url: str,
    token: str,
    endpoint: str,
    payload: dict,
    timeout: int = 30,
) -> dict:
    """POST *payload* to a SiYuan API *endpoint* and return the ``data`` field.

    Args:
        base_url: SiYuan instance URL, e.g. ``"http://127.0.0.1:6806"``.
        token: SiYuan API token.
        endpoint: API path, e.g. ``"/api/filetree/createDocWithMd"``.
        payload: JSON-serialisable request body.
        timeout: Request timeout in seconds (default 30).

    Returns:
        The ``data`` field of the SiYuan response (empty dict when absent).

    Raises:
        RuntimeError: When the SiYuan API returns a non-zero ``code``.
        requests.RequestException: On network / HTTP failures.
    """
    resp = requests.post(
        f"{base_url.rstrip('/')}{endpoint}",
        json=payload,
        headers={"Content-Type": "application/json", "Authorization": f"Token {token}"},
        timeout=timeout,
    )
    resp.raise_for_status()
    body = resp.json()
    if body.get("code", 0) != 0:
        raise RuntimeError(f"SiYuan error [{body.get('code')}]: {body.get('msg')}")
    return body.get("data") or {}


def infer_title(content: str, fallback: str) -> str:
    """Infer a document title from Markdown content.

    Returns the text of the first ``# Heading``, or the first non-empty
    non-heading line (truncated to 80 chars), or *fallback* if content is empty.
    """
    first_text: str | None = None
    for line in content.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("# "):
            return stripped[2:].strip()
        if first_text is None and not stripped.startswith("#"):
            first_text = stripped[:80]
    return first_text or fallback


def resolve_path(template: str, session_id: str, title: Optional[str]) -> str:
    """Fill date/session/title placeholders in a SiYuan path template.

    Supported placeholders: ``{session_id}``, ``{title}``, ``{date}``,
    ``{year}``, ``{month}``, ``{day}``.

    ``{title}`` is slugified (lowercased, non-word chars → hyphens).
    Falls back to *session_id* when *title* is absent.
    """
    now = datetime.now(timezone.utc)
    slug = re.sub(r"[^\w\-]", "-", (title or session_id).lower()).strip("-")
    return template.format(
        session_id=session_id,
        title=slug,
        date=now.strftime("%Y-%m-%d"),
        year=now.strftime("%Y"),
        month=now.strftime("%m"),
        day=now.strftime("%d"),
    )
