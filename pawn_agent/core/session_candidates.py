"""Structured session candidate helpers for agent session selection."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
import re
from typing import Any

from pawn_agent.core.chat_primitives import normalize_output


@dataclass(frozen=True)
class SessionCandidate:
    session_id: str
    title: str | None = None
    updated_at: datetime | None = None
    created_at: datetime | None = None
    summary: str | None = None
    segments: int | None = None
    duration_seconds: float | None = None


def _coerce_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value
    if value in (None, ""):
        return None
    text = normalize_output(value).strip()
    if not text or text.lower() == "unknown":
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            pass
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def _coerce_int(value: Any) -> int | None:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _coerce_float(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def candidate_to_state(candidate: SessionCandidate) -> dict[str, Any]:
    return {
        "session_id": candidate.session_id,
        "title": candidate.title,
        "updated_at": candidate.updated_at.isoformat() if candidate.updated_at else None,
        "created_at": candidate.created_at.isoformat() if candidate.created_at else None,
        "summary": candidate.summary,
        "segments": candidate.segments,
        "duration_seconds": candidate.duration_seconds,
    }


def candidate_from_state(value: Mapping[str, object]) -> SessionCandidate | None:
    session_id = normalize_output(value.get("session_id", "")).strip()
    if not session_id:
        return None
    return SessionCandidate(
        session_id=session_id,
        title=normalize_output(value.get("title")).strip() or None,
        updated_at=_coerce_datetime(value.get("updated_at")),
        created_at=_coerce_datetime(value.get("created_at")),
        summary=normalize_output(value.get("summary")).strip() or None,
        segments=_coerce_int(value.get("segments")),
        duration_seconds=_coerce_float(value.get("duration_seconds")),
    )


def candidates_to_state(candidates: Iterable[SessionCandidate]) -> list[dict[str, Any]]:
    return [candidate_to_state(candidate) for candidate in candidates]


def candidates_from_state(value: Any) -> list[SessionCandidate]:
    if not isinstance(value, list):
        return []
    candidates: list[SessionCandidate] = []
    for item in value:
        if not isinstance(item, Mapping):
            continue
        candidate = candidate_from_state(item)
        if candidate is not None:
            candidates.append(candidate)
    return candidates


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return "unknown"
    total = int(max(seconds, 0))
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    return f"{m}m {s:02d}s"


def render_session_catalog(candidates: Iterable[SessionCandidate]) -> str:
    items = list(candidates)
    if not items:
        return "No sessions found in the database."
    lines = [f"Found {len(items)} session(s):"]
    for candidate in items:
        updated = (
            candidate.updated_at.strftime("%Y-%m-%d %H:%M") if candidate.updated_at else "unknown"
        )
        segments = candidate.segments if candidate.segments is not None else "unknown"
        line = (
            f"  {candidate.session_id}"
            f"  |  segments: {segments}"
            f"  |  duration: {_fmt_duration(candidate.duration_seconds)}"
            f"  |  updated: {updated}"
        )
        if candidate.title:
            line += f"  |  title: {candidate.title}"
        lines.append(line)
    return "\n".join(lines)


_SEGMENTS_RE = re.compile(r"\bsegments:\s*(\d+)\b", re.IGNORECASE)
_UPDATED_RE = re.compile(r"\bupdated:\s*([^|]+)", re.IGNORECASE)


def parse_session_candidates_from_list_output(tool_output: str) -> list[SessionCandidate]:
    candidates: list[SessionCandidate] = []
    for raw_line in normalize_output(tool_output).splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("found "):
            continue
        session_id = line.split("|", 1)[0].strip()
        if not session_id or session_id.lower().startswith(("no sessions", "error ")):
            continue
        segments_match = _SEGMENTS_RE.search(line)
        updated_match = _UPDATED_RE.search(line)
        candidates.append(
            SessionCandidate(
                session_id=session_id,
                segments=int(segments_match.group(1)) if segments_match else None,
                updated_at=(
                    _coerce_datetime(updated_match.group(1).strip()) if updated_match else None
                ),
            )
        )
    return candidates


def session_search_text(candidate: SessionCandidate) -> str:
    return "\n".join(
        part
        for part in [
            candidate.session_id,
            candidate.title,
            candidate.summary,
            candidate.updated_at.isoformat() if candidate.updated_at else None,
        ]
        if part
    )
