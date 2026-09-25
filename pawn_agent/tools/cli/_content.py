"""Shared --content / --content-file guards for note CliTools."""

from __future__ import annotations

from pathlib import Path

_CONTENT_MAX_CHARS = 240
_CONTENT_REJECT_MSG = (
    "Error: --content is limited to short one-line text "
    f"(max {_CONTENT_MAX_CHARS} chars, no newlines). "
    "For Markdown use --content-file @note with a ```file note block."
)


def reject_unsafe_inline_content(content: str) -> str | None:
    """Return an error message when *content* is too long or multiline."""
    if "\n" in content or "\r" in content:
        return _CONTENT_REJECT_MSG
    if len(content) > _CONTENT_MAX_CHARS:
        return _CONTENT_REJECT_MSG
    return None


def read_content_file(path: str) -> str:
    """Read UTF-8 Markdown from a filesystem path (``@note`` resolved by sallm)."""
    return Path(path).expanduser().read_text(encoding="utf-8")
