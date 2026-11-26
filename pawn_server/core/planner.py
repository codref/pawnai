"""Day planner — vault event atoms, day merge, session link suggestions.

Events live as ``pawn: event`` notes under ``planner.events_dir``. Prep text
is rewritten only by the ``meeting_prep`` job / ``event_prep_update`` tool.
"""

from __future__ import annotations

import hashlib
import logging
import re
from datetime import date, datetime, timezone
from typing import Any, Optional
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from pawn_agent.tools.ideas_impl import idea_filename_title
from pawn_core.goals import slugify
from pawn_core.vault import (
    VaultNotFound,
    VaultWriteDenied,
    dump_frontmatter,
    normalize_vault_key,
    parse_frontmatter,
)
from pawn_core.vault_config import vault_store_from_config
from pawn_server.core.vault_events import publish_vault_event

logger = logging.getLogger(__name__)

_PREP_SECTION_RE = re.compile(
    r"(?is)^##\s*Prep\s*\n+(.*?)(?=^##\s|\Z)",
    re.MULTILINE,
)
_TOKEN_RE = re.compile(r"[a-z0-9]+")
_LOCAL_DT_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2})(?:[T ](\d{2}:\d{2})(?::\d{2})?)?$"
)


class PlannerError(Exception):
    """Client-facing planner failure with an HTTP status."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _agent_root(cfg: Any) -> str:
    root = getattr(getattr(cfg, "vault", None), "agent_root", None) or "Pawn"
    return str(root).strip().strip("/") or "Pawn"


def events_dir(cfg: Any) -> str:
    section = getattr(cfg, "planner", None)
    raw = str(getattr(section, "events_dir", "") or "").strip() if section else ""
    if not raw:
        raw = "{agent_root}/Events"
    return normalize_vault_key(
        raw.replace("{agent_root}", _agent_root(cfg))
    ).rstrip("/")


def _timezone_name(cfg: Any) -> str:
    return (getattr(getattr(cfg, "coworker", None), "timezone", "") or "UTC").strip() or "UTC"


def _today(cfg: Any) -> str:
    try:
        return datetime.now(ZoneInfo(_timezone_name(cfg))).date().isoformat()
    except ZoneInfoNotFoundError:
        return datetime.now(timezone.utc).date().isoformat()


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _meta_str(value: Any) -> str:
    if value is None:
        return ""
    if hasattr(value, "isoformat"):
        try:
            return str(value.isoformat())
        except Exception:
            return str(value).strip()
    return str(value).strip()


def _meta_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(x).strip() for x in value if str(x).strip()]
    text = str(value).strip()
    if not text:
        return []
    if "," in text:
        return [p.strip() for p in text.split(",") if p.strip()]
    return [text]


def _meta_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value or "").strip().lower()
    return text in {"1", "true", "yes", "on"}


def _meta_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def day_from_local(value: str) -> str:
    """Extract ``YYYY-MM-DD`` from a local start/end string."""
    raw = (value or "").strip()
    match = _LOCAL_DT_RE.match(raw)
    if match:
        return match.group(1)
    if len(raw) >= 10 and raw[4] == "-" and raw[7] == "-":
        return raw[:10]
    return ""


_US_DT_RE = re.compile(
    r"^(\d{1,2})/(\d{1,2})/(\d{4}),?\s+(\d{1,2}):(\d{2})\s*([AP]M)\s*$",
    re.I,
)


def normalize_local_dt(value: str) -> str:
    """Normalize to ``YYYY-MM-DD`` or ``YYYY-MM-DDTHH:MM`` (no TZ)."""
    raw = (value or "").strip()
    us = _US_DT_RE.match(raw)
    if us:
        month, day, year = int(us.group(1)), int(us.group(2)), int(us.group(3))
        hour, minute = int(us.group(4)), int(us.group(5))
        ap = us.group(6).upper()
        if ap == "PM" and hour < 12:
            hour += 12
        if ap == "AM" and hour == 12:
            hour = 0
        return f"{year:04d}-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}"
    raw = raw.replace(" ", "T")
    match = _LOCAL_DT_RE.match(raw)
    if not match:
        return raw
    day, hm = match.group(1), match.group(2)
    if hm:
        return f"{day}T{hm}"
    return day


def _normalize_identity_url(url: str) -> str:
    """Lowercase host, drop query/fragment, trim trailing slash (keep root as '')."""
    raw = (url or "").strip()
    if not raw:
        return ""
    parts = urlsplit(raw)
    if not parts.scheme or not parts.netloc:
        return raw.rstrip("/")
    path = (parts.path or "").rstrip("/")
    return urlunsplit((parts.scheme.lower(), parts.netloc.lower(), path, "", ""))


def event_identity_url(url: str) -> str:
    """Return a normalized URL safe for fingerprinting, or ``\"\"`` if too generic.

    Teams/Outlook SPA shells (``https://teams.cloud.microsoft/``, calendar
    week views, …) are shared across unrelated meetings and must not key
    upserts. Prefer join links (``/meet/…``, Zoom ``/j/…``) and Outlook
    item deeplinks.
    """
    norm = _normalize_identity_url(url)
    if not norm:
        return ""
    parts = urlsplit(norm if "://" in norm else f"https://{norm}")
    host = (parts.netloc or "").lower()
    path = parts.path or ""
    if not host and "://" not in norm:
        # Non-URL token — not an identity key.
        return ""

    teams_hosts = (
        "teams.cloud.microsoft",
        "teams.microsoft.com",
        "teams.live.com",
    )
    if host in teams_hosts or host.endswith(".teams.microsoft.com"):
        if re.search(r"/(?:meet|meetup-join|l/meetup-join)/", path, re.I):
            return _normalize_identity_url(norm)
        return ""

    if "outlook." in host or host.endswith(".office.com") or host in (
        "office.com",
        "office365.com",
    ):
        if re.search(r"/(?:item|deeplink)/", path, re.I):
            return _normalize_identity_url(norm)
        return ""

    if "zoom.us" in host or host.endswith("zoom.com"):
        if re.search(r"/(?:j|meeting)/", path, re.I):
            return _normalize_identity_url(norm)
        return ""

    # Other hosts: any non-root path is specific enough.
    if path not in ("", "/"):
        return _normalize_identity_url(norm)
    return ""


def event_fingerprint(
    *,
    source_url: str = "",
    meeting_url: str = "",
    title: str = "",
    start: str = "",
) -> str:
    """Stable upsert key: meeting join URL, then specific page URL, else title|start.

    Generic app-shell URLs are ignored so unrelated Teams clips do not collapse
    into one event note.
    """
    meet = event_identity_url(meeting_url)
    if meet:
        return hashlib.sha256(f"meet:{meet}".encode("utf-8")).hexdigest()[:24]
    page = event_identity_url(source_url)
    if page:
        return hashlib.sha256(f"url:{page}".encode("utf-8")).hexdigest()[:24]
    key = f"title:{(title or '').strip().lower()}|{normalize_local_dt(start)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


def event_identities_conflict(
    existing: dict[str, Any],
    *,
    source_url: str = "",
    meeting_url: str = "",
    title: str = "",
    start: str = "",
) -> bool:
    """True when a sticky path/id points at a different meeting than this capture."""
    old_meet = event_identity_url(str(existing.get("meeting_url") or ""))
    new_meet = event_identity_url(meeting_url)
    if old_meet and new_meet and old_meet != new_meet:
        return True
    old_src = event_identity_url(str(existing.get("source_url") or ""))
    new_src = event_identity_url(source_url)
    if old_src and new_src and old_src != new_src:
        return True
    if old_meet or new_meet or old_src or new_src:
        # At least one side has a strong URL; only URL mismatch is a conflict.
        return False
    old_title = str(existing.get("title") or "").strip().lower()
    new_title = (title or "").strip().lower()
    old_start = normalize_local_dt(str(existing.get("start") or ""))
    new_start = normalize_local_dt(start) if start else ""
    if old_title and new_title and old_start and new_start:
        return old_title != new_title or old_start != new_start
    return False


def render_event_note(
    *,
    event_id: str,
    title: str,
    start: str = "",
    end: str = "",
    source: str = "manual",
    source_url: str = "",
    attendees: list[str] | None = None,
    topics: list[str] | None = None,
    organizer: str = "",
    location: str = "",
    meeting_url: str = "",
    session_id: str = "",
    fingerprint: str = "",
    prep_job_id: str = "",
    prep_at: str = "",
    focus: bool = False,
    focus_before: int = 0,
    body: str = "",
    prep: str = "",
) -> str:
    """Serialize one ``pawn: event`` note."""
    heading = idea_filename_title(title) or "event"
    meta: dict[str, Any] = {
        "pawn": "event",
        "id": event_id,
        "title": heading,
        "start": normalize_local_dt(start) if start else "",
        "end": normalize_local_dt(end) if end else "",
        "source": (source or "manual").strip() or "manual",
        "fingerprint": fingerprint
        or event_fingerprint(
            source_url=source_url,
            meeting_url=meeting_url,
            title=heading,
            start=start,
        ),
    }
    if source_url.strip():
        meta["source_url"] = source_url.strip()
    if attendees:
        meta["attendees"] = [a for a in attendees if a]
    if topics:
        meta["topics"] = [t for t in topics if t]
    if organizer.strip():
        meta["organizer"] = organizer.strip()
    if location.strip():
        meta["location"] = location.strip()
    if meeting_url.strip():
        meta["meeting_url"] = meeting_url.strip()
    if session_id.strip():
        meta["session_id"] = session_id.strip()
    if prep_job_id.strip():
        meta["prep_job_id"] = prep_job_id.strip()
    if prep_at.strip():
        meta["prep_at"] = prep_at.strip()
    if focus:
        meta["focus"] = True
        meta["focus_before"] = max(0, int(focus_before or 0)) or 15
    elif focus_before:
        meta["focus_before"] = max(0, int(focus_before))

    parts = [f"# {heading}", ""]
    body_text = (body or "").strip()
    if body_text:
        # Drop a leading heading that duplicates the title.
        lines = body_text.splitlines()
        if lines and lines[0].lstrip().startswith("#"):
            lines = lines[1:]
            while lines and not lines[0].strip():
                lines = lines[1:]
            body_text = "\n".join(lines).strip()
    if body_text:
        # Strip an existing ## Prep so we control that section.
        body_text = _PREP_SECTION_RE.sub("", body_text).strip()
    if body_text:
        parts.append(body_text)
        parts.append("")
    prep_text = (prep or "").strip()
    if prep_text:
        parts.append("## Prep")
        parts.append("")
        parts.append(prep_text)
        parts.append("")
    return dump_frontmatter(meta, "\n".join(parts))


_SNIPPET_MARK_RE = re.compile(r"<!--\s*pawn-snippet:([A-Za-z0-9_-]+)\s*-->")
_IMAGE_EMBED_RE = re.compile(r"!\[\[([^\]]+)\]\]")
_STUB_TITLE_RE = re.compile(
    r"(?ix)^(?:\(\d+\)\s*)?(?:calendar|mail|outlook|microsoft\s*teams|"
    r"tracking|accepted|declined|busy|free|tentative|reminder|details|"
    r"chat|join|meeting|inbox|day|week|month)\b"
    r"|calendar\s*[|–—-]\s*microsoft\s*teams"
    r"|-\s*outlook\s*$"
    r"|^\s*outlook\s*event\s*$"
    r"|^\s*prepare for this meeting\s*$"
    r"|^\s*microsoft teams meeting\s*$"
)
_BREADCRUMB_SUBJECT_RE = re.compile(
    r"^(.+?)\s*[–—|-]\s*Meeting\s*[–—|-]\s*Calendar\b",
    re.I,
)


def is_stub_event_title(title: str) -> bool:
    """True for browser-tab / shell titles that are not a meeting subject."""
    text = (title or "").strip()
    if not text:
        return True
    if _STUB_TITLE_RE.search(text):
        return True
    # Bare "Calendar | Microsoft Teams" / unread-count prefixes.
    if re.match(r"^\(\d+\)\s*calendar\b", text, re.I):
        return True
    return False


def subject_from_page_title(raw: str) -> str:
    """Pull subject from ``Review … - Meeting - Calendar`` chrome titles."""
    text = (raw or "").strip()
    match = _BREADCRUMB_SUBJECT_RE.match(text)
    if match:
        sub = match.group(1).strip()
        return "" if is_stub_event_title(sub) else sub
    return ""


_MEETING_URL_RE = re.compile(
    r"https?://(?:teams\.microsoft\.com|teams\.live\.com|meet\.google\.com|"
    r"[\w.-]*zoom\.us)/\S+",
    re.I,
)
_DATE_RANGE_RE = re.compile(
    r"(\d{1,2}/\d{1,2}/\d{4}),?\s+(\d{1,2}:\d{2}\s*[AP]M)\s*[–—-]\s*"
    r"(\d{1,2}:\d{2}\s*[AP]M)",
    re.I,
)
_PERSON_ROLE_RE = re.compile(
    r"(?m)^(.{2,80}?)\s*\((Organizer|Required|Optional|Accepted|Tentative)\)\s*$",
    re.I,
)
_JUNK_PERSON = {
    "required",
    "optional",
    "organizer",
    "attendees",
    "participants",
    "accepted",
    "tentative",
    "invite",
    "calendar",
    "chat",
    "join",
    "copy",
    "delete",
    "archive",
    "reply",
    "forward",
    "busy",
    "reminder",
    "categorize",
    "private",
    "prepare",
    "copilot",
    "viva",
    "tracking",
    "meeting",
    "more",
    "close",
    "cancel",
    "save",
    "send",
    "help",
    "you",
    "me",
}
_JUNK_PERSON_RE = re.compile(r"\[(?:image|object)[^\]]*\]", re.I)


def _split_people(raw: str) -> list[str]:
    parts = re.split(r"[,;|/]| and ", raw or "")
    out: list[str] = []
    for p in parts:
        name = _JUNK_PERSON_RE.sub("", p)
        name = re.sub(r"\s*\([^)]*\)\s*$", "", name).strip(" \t.-")
        if not name or len(name) < 2 or len(name) > 80:
            continue
        if name.lower() in _JUNK_PERSON:
            continue
        if name.lower().startswith(("http://", "https://", "mailto:")):
            continue
        if not re.search(r"[A-Za-zÀ-ÿ]", name):
            continue
        # Toolbar labels are short lowercase singles; allow "Ada" / emails.
        if " " not in name and "@" not in name:
            if not re.match(r"^[A-ZÀ-Ÿ]", name):
                continue
            if len(name) < 2:
                continue
        out.append(name)
    return out


def merge_people(*lists: list[str] | None) -> list[str]:
    """Dedupe people names case-insensitively, preserving first spelling."""
    seen: set[str] = set()
    out: list[str] = []
    for lst in lists:
        for raw in lst or []:
            for name in _split_people(str(raw)):
                key = name.lower()
                if key in seen:
                    continue
                seen.add(key)
                out.append(name)
    return out[:40]


def title_from_event_content(*chunks: str) -> str:
    """Pick a meeting subject from captured text; empty when nothing usable."""
    candidates: list[str] = []
    skip_prefixes = (
        "source:",
        "attendees:",
        "participants:",
        "organizer:",
        "location:",
        "where:",
        "meeting:",
        "join:",
        "start:",
        "end:",
        "when:",
        "topics:",
        "tags:",
        "http://",
        "https://",
        "### snippet",
        "microsoft teams meeting",
        "meeting id:",
        "passcode:",
        "join the meeting",
        "need help?",
    )
    for chunk in chunks:
        crumb = subject_from_page_title(chunk or "")
        if crumb:
            candidates.append(crumb)
        for raw in (chunk or "").splitlines():
            line = raw.strip().lstrip("#").strip()
            if not line:
                continue
            lower = line.lower()
            if lower.startswith("outlook event:"):
                line = line.split(":", 1)[1].strip()
                lower = line.lower()
            if lower.startswith("subject:"):
                line = line.split(":", 1)[1].strip()
                lower = line.lower()
            crumb_line = subject_from_page_title(line)
            if crumb_line:
                candidates.append(crumb_line)
                continue
            if any(lower.startswith(p) for p in skip_prefixes):
                continue
            if _PERSON_ROLE_RE.match(line):
                continue
            if is_stub_event_title(line):
                continue
            # Date / time rows are not subjects.
            if re.match(r"^\d{1,2}/\d{1,2}/\d{4}", line):
                continue
            if re.search(r"\d{1,2}:\d{2}\s*[AP]M", line, re.I):
                continue
            if re.match(r"^(?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)\w*\b", line, re.I):
                continue
            if 2 <= len(line) <= 120:
                candidates.append(line)
    if not candidates:
        return ""
    # Prefer longer real subjects over short chrome words ("Tracking").
    candidates.sort(key=lambda s: (-len(s), s))
    for cand in candidates:
        if not is_stub_event_title(cand) and len(cand) >= 8:
            return idea_filename_title(cand)[:80]
    return idea_filename_title(candidates[0])[:80]


def enrich_event_fields_from_content(*chunks: str) -> dict[str, Any]:
    """Pull title/attendees/times/location/meeting URL from invite-like text."""
    blob = "\n".join(c for c in chunks if c)
    out: dict[str, Any] = {
        "title": title_from_event_content(*chunks),
        "attendees": [],
        "topics": [],
        "organizer": "",
        "location": "",
        "meeting_url": "",
        "start": "",
        "end": "",
    }
    if not blob.strip():
        return out

    people: list[str] = []
    for m in _PERSON_ROLE_RE.finditer(blob):
        name = m.group(1).strip()
        role = m.group(2).lower()
        if role == "organizer" and not out["organizer"]:
            out["organizer"] = name
        people.append(name)

    for label in ("attendees", "participants", "required attendees", "invitees"):
        m = re.search(rf"(?im)^{re.escape(label)}\s*:\s*(.+)$", blob)
        if m:
            people.extend(_split_people(m.group(1)))

    org = re.search(r"(?im)^(?:organizer|from|hosted by)\s*:\s*(.+)$", blob)
    if org and not out["organizer"]:
        out["organizer"] = _split_people(org.group(1))[0] if _split_people(org.group(1)) else org.group(1).strip()
        people.append(out["organizer"])

    out["attendees"] = merge_people(people)

    loc = re.search(r"(?im)^(?:location|where|room)\s*:\s*(.+)$", blob)
    if loc:
        out["location"] = loc.group(1).strip()[:160]

    meet = _MEETING_URL_RE.search(blob)
    if meet:
        out["meeting_url"] = meet.group(0).rstrip(").,;")

    for label, key in (("start", "start"), ("end", "end"), ("when", "start")):
        m = re.search(rf"(?im)^{label}\s*:\s*(.+)$", blob)
        if m and not out[key]:
            out[key] = m.group(1).strip()[:40]

    rng = _DATE_RANGE_RE.search(blob)
    if rng:
        if not out["start"]:
            out["start"] = f"{rng.group(1)}, {rng.group(2)}"
        if not out["end"]:
            out["end"] = f"{rng.group(1)}, {rng.group(3)}"

    for label in ("topics", "tags", "about"):
        m = re.search(rf"(?im)^{label}\s*:\s*(.+)$", blob)
        if m:
            out["topics"] = [
                t.strip() for t in re.split(r"[,;|]", m.group(1)) if t.strip()
            ][:12]
            break

    return out


def parse_event_note(text: str, *, path: str = "") -> dict[str, Any]:
    """Parse a ``pawn: event`` note. Raises ValueError when it is not one."""
    meta, body = parse_frontmatter(text or "")
    if str(meta.get("pawn") or "").strip() != "event":
        raise ValueError("frontmatter pawn must be 'event'")
    prep = ""
    match = _PREP_SECTION_RE.search(body)
    if match:
        prep = match.group(1).strip()
        rest = _PREP_SECTION_RE.sub("", body).strip()
    else:
        rest = body.strip()
    title = _meta_str(meta.get("title")) or idea_filename_title(rest.splitlines()[0] if rest else "event")
    snippet_ids = _SNIPPET_MARK_RE.findall(rest)
    image_embeds = _IMAGE_EMBED_RE.findall(rest)
    return {
        "id": _meta_str(meta.get("id")),
        "title": title,
        "start": normalize_local_dt(_meta_str(meta.get("start"))),
        "end": normalize_local_dt(_meta_str(meta.get("end"))),
        "source": _meta_str(meta.get("source")) or "manual",
        "source_url": _meta_str(meta.get("source_url")),
        "attendees": _meta_list(meta.get("attendees")),
        "topics": _meta_list(meta.get("topics")),
        "organizer": _meta_str(meta.get("organizer")),
        "location": _meta_str(meta.get("location")),
        "meeting_url": _meta_str(meta.get("meeting_url")),
        "session_id": _meta_str(meta.get("session_id")),
        "fingerprint": _meta_str(meta.get("fingerprint")),
        "prep_job_id": _meta_str(meta.get("prep_job_id")),
        "prep_at": _meta_str(meta.get("prep_at")),
        "focus": _meta_bool(meta.get("focus")),
        "focus_before": _meta_int(meta.get("focus_before"), 0),
        "path": path,
        "body": rest,
        "prep": prep,
        "day": day_from_local(_meta_str(meta.get("start"))) or "",
        "clip_count": len(snippet_ids),
        "image_count": len(image_embeds),
        "has_prep": bool(prep.strip() or _meta_str(meta.get("prep_at"))),
    }


def replace_prep_section(text: str, prep: str, *, prep_job_id: str = "") -> str:
    """Replace or append ``## Prep`` and update prep frontmatter fields."""
    meta, body = parse_frontmatter(text or "")
    if str(meta.get("pawn") or "").strip() != "event":
        raise ValueError("frontmatter pawn must be 'event'")
    prep_body = (prep or "").rstrip() + "\n"
    match = _PREP_SECTION_RE.search(body)
    if match:
        start, end = match.span(1)
        # Keep heading; replace body after "## Prep\n"
        heading_end = match.start(1)
        new_body = body[:heading_end] + prep_body + body[end:]
    else:
        base = body.rstrip()
        new_body = (base + "\n\n" if base else "") + f"## Prep\n\n{prep_body}"
    meta["prep_at"] = _now_iso()
    if prep_job_id.strip():
        meta["prep_job_id"] = prep_job_id.strip()
    return dump_frontmatter(meta, new_body)


def extract_prep(text: str) -> str:
    match = _PREP_SECTION_RE.search(text or "")
    return match.group(1).strip() if match else ""


def _event_path(cfg: Any, *, day: str, title: str, short_id: str) -> str:
    stem = slugify(idea_filename_title(title))[:48] or "event"
    day_part = day or _today(cfg)
    return f"{events_dir(cfg)}/{day_part}-{stem}-{short_id}.md"


def _list_event_keys(cfg: Any, store: Any) -> list[str]:
    folder = events_dir(cfg)
    try:
        return store.list(folder, suffix=".md")
    except Exception as exc:  # noqa: BLE001
        logger.debug("list events failed: %s", exc)
        return []


def _load_events(cfg: Any, store: Any) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in _list_event_keys(cfg, store):
        try:
            text = store.read(key)
            out.append(parse_event_note(text, path=key))
        except Exception as exc:  # noqa: BLE001
            logger.debug("skip event %s: %s", key, exc)
    return out


def find_event_by_id(cfg: Any, event_id: str, *, store: Any = None) -> dict[str, Any]:
    if store is None:
        store = vault_store_from_config(cfg)
    needle = (event_id or "").strip()
    if not needle:
        raise PlannerError("event id is required", status_code=422)
    for ev in _load_events(cfg, store):
        if ev.get("id") == needle or ev.get("path", "").endswith(f"-{needle}.md"):
            return ev
    raise PlannerError(f"event not found: {needle}", status_code=404)


def find_event_by_fingerprint(
    cfg: Any, fingerprint: str, *, store: Any = None
) -> Optional[dict[str, Any]]:
    if store is None:
        store = vault_store_from_config(cfg)
    fp = (fingerprint or "").strip()
    if not fp:
        return None
    for ev in _load_events(cfg, store):
        if ev.get("fingerprint") == fp:
            return ev
        # Recompute from fields so notes stamped with an older algorithm still
        # match when meeting_url / specific source_url / title|start agree.
        recomputed = event_fingerprint(
            source_url=str(ev.get("source_url") or ""),
            meeting_url=str(ev.get("meeting_url") or ""),
            title=str(ev.get("title") or ""),
            start=str(ev.get("start") or ""),
        )
        if recomputed == fp:
            return ev
    return None


def upsert_event(
    cfg: Any,
    *,
    title: str,
    start: str = "",
    end: str = "",
    source: str = "manual",
    source_url: str = "",
    attendees: list[str] | None = None,
    topics: list[str] | None = None,
    organizer: str = "",
    location: str = "",
    meeting_url: str = "",
    session_id: str = "",
    body: str = "",
    event_id: str = "",
    store: Any = None,
) -> dict[str, Any]:
    """Create or update an event note (fingerprint upsert)."""
    if store is None:
        store = vault_store_from_config(cfg)
    heading = idea_filename_title(title or "").strip()
    if not heading:
        raise PlannerError("title is required", status_code=422)
    start_n = normalize_local_dt(start) if start else ""
    end_n = normalize_local_dt(end) if end else ""
    if not start_n:
        start_n = _today(cfg)
    fp = event_fingerprint(
        source_url=source_url,
        meeting_url=meeting_url,
        title=heading,
        start=start_n,
    )
    existing = find_event_by_fingerprint(cfg, fp, store=store)
    if existing is None and event_id.strip():
        try:
            existing = find_event_by_id(cfg, event_id, store=store)
        except PlannerError:
            existing = None
        if existing is not None and event_identities_conflict(
            existing,
            source_url=source_url,
            meeting_url=meeting_url,
            title=heading,
            start=start_n,
        ):
            existing = None

    if existing is not None:
        eid = existing["id"]
        path = existing["path"]
        try:
            prev = store.read(path)
            prev_meta, _ = parse_frontmatter(prev)
            prep = extract_prep(prev)
            prep_job_id = _meta_str(prev_meta.get("prep_job_id"))
            prep_at = _meta_str(prev_meta.get("prep_at"))
            linked = session_id.strip() or _meta_str(prev_meta.get("session_id"))
            prev_org = _meta_str(prev_meta.get("organizer"))
            prev_loc = _meta_str(prev_meta.get("location"))
            prev_meet = _meta_str(prev_meta.get("meeting_url"))
            prev_focus = _meta_bool(prev_meta.get("focus"))
            prev_focus_before = _meta_int(prev_meta.get("focus_before"), 0)
        except VaultNotFound:
            prep = existing.get("prep") or ""
            prep_job_id = existing.get("prep_job_id") or ""
            prep_at = existing.get("prep_at") or ""
            linked = session_id.strip() or existing.get("session_id") or ""
            prev_org = existing.get("organizer") or ""
            prev_loc = existing.get("location") or ""
            prev_meet = existing.get("meeting_url") or ""
            prev_focus = bool(existing.get("focus"))
            prev_focus_before = int(existing.get("focus_before") or 0)
        merged_meet = meeting_url.strip() or prev_meet
        # Recompute after merging meeting_url so a later join-link upgrades the key.
        fp = event_fingerprint(
            source_url=source_url or existing.get("source_url") or "",
            meeting_url=merged_meet,
            title=heading,
            start=start_n,
        )
        text = render_event_note(
            event_id=eid,
            title=heading,
            start=start_n,
            end=end_n or existing.get("end") or "",
            source=source or existing.get("source") or "manual",
            source_url=source_url or existing.get("source_url") or "",
            attendees=attendees if attendees is not None else existing.get("attendees"),
            topics=topics if topics is not None else existing.get("topics"),
            organizer=organizer.strip() or prev_org,
            location=location.strip() or prev_loc,
            meeting_url=merged_meet,
            session_id=linked,
            fingerprint=fp,
            prep_job_id=prep_job_id,
            prep_at=prep_at,
            focus=prev_focus,
            focus_before=prev_focus_before,
            body=body if body.strip() else existing.get("body") or "",
            prep=prep,
        )
        day = day_from_local(start_n) or _today(cfg)
        filename = str(path).rsplit("/", 1)[-1]
        old_day = (
            filename[:10]
            if len(filename) >= 10 and filename[4] == "-" and filename[7] == "-"
            else ""
        )
        new_path = path
        if old_day and old_day != day:
            # Keep the filename stem stable across title edits; only move when
            # the start day changes so Day view's prefix filter can find it.
            short_id = eid[4:] if str(eid).startswith("evt_") and len(eid) > 4 else ""
            if not short_id:
                stem = filename.removesuffix(".md")
                short_id = stem.rsplit("-", 1)[-1] if "-" in stem else uuid4().hex[:8]
            new_path = _event_path(cfg, day=day, title=heading, short_id=short_id)
        try:
            store.write(new_path, text)
            if new_path != path:
                try:
                    store.delete(path)
                except VaultNotFound:
                    pass
                publish_vault_event([new_path, path], source="planner")
            else:
                publish_vault_event([path], source="planner")
        except VaultWriteDenied as exc:
            raise PlannerError(str(exc), status_code=403) from exc
        return parse_event_note(text, path=new_path)

    short_id = uuid4().hex[:8]
    eid = (event_id or "").strip() or f"evt_{short_id}"
    if eid.startswith("evt_") and len(eid) > 4:
        short_id = eid[4:]
    day = day_from_local(start_n) or _today(cfg)
    path = _event_path(cfg, day=day, title=heading, short_id=short_id)
    text = render_event_note(
        event_id=eid,
        title=heading,
        start=start_n,
        end=end_n,
        source=source or "manual",
        source_url=source_url,
        attendees=attendees,
        topics=topics,
        organizer=organizer,
        location=location,
        meeting_url=meeting_url,
        session_id=session_id,
        fingerprint=fp,
        body=body,
    )
    try:
        store.write(path, text)
    except VaultWriteDenied as exc:
        raise PlannerError(str(exc), status_code=403) from exc
    publish_vault_event([path], source="planner")
    return parse_event_note(text, path=path)


def event_assets_dir(cfg: Any) -> str:
    return f"{events_dir(cfg)}/assets"


def _append_snippets_to_event(
    cfg: Any,
    *,
    path: str,
    snippets: list[dict[str, Any]],
    store: Any,
) -> tuple[list[str], list[str]]:
    """Append text/image snippet blocks (idempotent via pawn-snippet markers)."""
    from pawn_server.core.captures import (  # noqa: PLC0415
        CaptureError,
        _SNIPPET_ID_RE,
        _content_type,
        _decode_image,
        _image_suffix,
        note_has_snippet,
        render_snippet_block,
    )

    try:
        existing = store.read(path)
    except VaultNotFound as exc:
        raise PlannerError(f"event note not found: {path}", status_code=404) from exc

    written: list[str] = []
    skipped: list[str] = []
    blocks: list[str] = []
    image_keys: list[str] = []

    for raw in snippets:
        snippet_id = str(raw.get("id") or "").strip()
        if not snippet_id or not _SNIPPET_ID_RE.match(snippet_id):
            raise PlannerError(f"invalid snippet id {snippet_id!r}", status_code=422)
        if note_has_snippet(existing, snippet_id) or note_has_snippet(
            "\n".join(blocks), snippet_id
        ):
            skipped.append(snippet_id)
            continue
        skind = str(raw.get("kind") or "text").strip().lower()
        if skind not in {"text", "image"}:
            raise PlannerError(f"unknown snippet kind {skind!r}", status_code=422)
        image_key = ""
        if skind == "image":
            try:
                data = _decode_image(str(raw.get("data_base64") or ""))
            except CaptureError as exc:
                raise PlannerError(str(exc), status_code=exc.status_code) from exc
            suffix = _image_suffix(str(raw.get("media_type") or ""))
            image_key = f"{event_assets_dir(cfg)}/{snippet_id}{suffix}"
            store.write_bytes(image_key, data, content_type=_content_type(suffix))
            image_keys.append(image_key)
        block = render_snippet_block(
            snippet_id=snippet_id,
            kind=skind,
            text=str(raw.get("text") or ""),
            image_key=image_key,
            source_url=str(raw.get("source_url") or ""),
            captured_at=str(raw.get("captured_at") or ""),
        )
        blocks.append(block)
        written.append(snippet_id)

    if not blocks:
        return written, skipped

    meta, body = parse_frontmatter(existing)
    prep = extract_prep(existing)
    rest = _PREP_SECTION_RE.sub("", body).rstrip()
    # Drop a leading H1 that duplicates the title — keep real content.
    new_body = rest + ("\n\n" if rest else "") + "".join(blocks)
    if prep.strip():
        new_body = new_body.rstrip() + f"\n\n## Prep\n\n{prep.strip()}\n"
    updated = dump_frontmatter(meta, new_body)
    try:
        store.write(path, updated)
    except VaultWriteDenied as exc:
        raise PlannerError(str(exc), status_code=403) from exc
    publish_vault_event([path, *image_keys], source="planner")
    return written, skipped


def upsert_event_from_capture(
    cfg: Any,
    *,
    target: dict[str, Any],
    snippets: list[dict[str, Any]],
    store: Any = None,
) -> dict[str, Any]:
    """Capture target ``kind: event`` → upsert metadata, then append snippets.

    Text and image snippets are stored under the event note (images in
    ``Events/assets/``). Re-saves are idempotent via snippet markers — a later
    whole-view clip updates the same event instead of leaving a stub.
    """
    if store is None:
        store = vault_store_from_config(cfg)
    title = str(target.get("title") or "").strip()
    start = str(target.get("start") or "").strip()
    end = str(target.get("end") or "").strip()
    source_url = str(target.get("source_url") or "").strip()
    attendees = _meta_list(target.get("attendees"))
    topics = _meta_list(target.get("topics"))
    organizer = str(target.get("organizer") or "").strip()
    location = str(target.get("location") or "").strip()
    meeting_url = str(target.get("meeting_url") or "").strip()
    path_hint = normalize_vault_key(str(target.get("path") or "").strip())
    event_id_hint = str(target.get("id") or target.get("event_id") or "").strip()
    text_chunks: list[str] = []

    for raw in snippets:
        if not source_url:
            source_url = str(raw.get("source_url") or "").strip()
        if str(raw.get("kind") or "text").strip().lower() == "text":
            chunk = str(raw.get("text") or "").strip()
            if chunk:
                text_chunks.append(chunk)

    existing_title = ""
    existing_attendees: list[str] = []
    existing_topics: list[str] = []
    sticky_existing: Optional[dict[str, Any]] = None
    if path_hint:
        # Sticky path from the extension — prefer appending to that note, but
        # only after enrichment confirms it is still the same meeting.
        try:
            sticky_existing = parse_event_note(store.read(path_hint), path=path_hint)
            event_id_hint = event_id_hint or str(sticky_existing.get("id") or "")
            existing_title = str(sticky_existing.get("title") or "")
            existing_attendees = list(sticky_existing.get("attendees") or [])
            existing_topics = list(sticky_existing.get("topics") or [])
            # Body text can also yield a subject on later clips.
            if sticky_existing.get("body"):
                text_chunks.append(str(sticky_existing.get("body") or ""))
        except Exception:  # noqa: BLE001
            sticky_existing = None

    enriched = enrich_event_fields_from_content(title, *text_chunks)
    # Content wins over browser-tab stubs ("Calendar | Microsoft Teams", …).
    # Do not fall back to sticky title yet — that waits until identity checks out.
    from_content = str(enriched.get("title") or "")
    if from_content:
        title = from_content
    if not title or is_stub_event_title(title):
        title = from_content or title

    if not start and enriched.get("start"):
        start = normalize_local_dt(str(enriched["start"])) or str(enriched["start"])
    if not end and enriched.get("end"):
        end = normalize_local_dt(str(enriched["end"])) or str(enriched["end"])
    if not organizer and enriched.get("organizer"):
        organizer = str(enriched["organizer"])
    if not location and enriched.get("location"):
        location = str(enriched["location"])
    if not meeting_url and enriched.get("meeting_url"):
        meeting_url = str(enriched["meeting_url"])
    attendees = merge_people(attendees, enriched.get("attendees"))
    topic_seen: set[str] = set()
    merged_topics: list[str] = []
    for lst in (topics, list(enriched.get("topics") or [])):
        for t in lst or []:
            key = str(t).strip().lower()
            if not key or key in topic_seen:
                continue
            topic_seen.add(key)
            merged_topics.append(str(t).strip())
    topics = merged_topics[:12]

    # Screenshot / whole-view clips: vision fills gaps DOM scrape missed.
    from pawn_server.core.event_vision import (  # noqa: PLC0415
        decode_snippet_images,
        extract_event_fields_from_images,
        needs_event_vision,
    )

    image_bytes = decode_snippet_images(snippets)
    if image_bytes and needs_event_vision(
        title=title or existing_title,
        start=start,
        attendees=attendees,
        organizer=organizer,
    ):
        vision = extract_event_fields_from_images(cfg, image_bytes)
        v_title = str(vision.get("title") or "").strip()
        v_title = subject_from_page_title(v_title) or v_title
        if (
            v_title
            and not is_stub_event_title(v_title)
            and (not title or is_stub_event_title(title))
        ):
            title = idea_filename_title(v_title)[:80]
        if not start and vision.get("start"):
            start = normalize_local_dt(str(vision["start"])) or str(vision["start"])
        if not end and vision.get("end"):
            end = normalize_local_dt(str(vision["end"])) or str(vision["end"])
        if not organizer and vision.get("organizer"):
            organizer = str(vision["organizer"]).strip()
        if not location and vision.get("location"):
            location = str(vision["location"]).strip()
        if not meeting_url and vision.get("meeting_url"):
            meeting_url = str(vision["meeting_url"]).strip()
        attendees = merge_people(attendees, vision.get("attendees"))
        for t in vision.get("topics") or []:
            key = str(t).strip().lower()
            if key and key not in topic_seen:
                topic_seen.add(key)
                topics.append(str(t).strip())
        topics = topics[:12]

    if sticky_existing is not None and event_identities_conflict(
        sticky_existing,
        source_url=source_url,
        meeting_url=meeting_url,
        title=title,
        start=start,
    ):
        # Different meeting than the sticky note — create a fresh event.
        path_hint = ""
        event_id_hint = ""
        sticky_existing = None
        existing_title = ""
        existing_attendees = []
        existing_topics = []
    elif sticky_existing is not None:
        # Same meeting: fill gaps from the sticky note.
        if not title or is_stub_event_title(title):
            if existing_title and not is_stub_event_title(existing_title):
                title = existing_title
            elif existing_title:
                title = existing_title
        if not start:
            start = str(sticky_existing.get("start") or "")
        if not end:
            end = str(sticky_existing.get("end") or "")
        if not source_url:
            source_url = str(sticky_existing.get("source_url") or "")
        if not organizer:
            organizer = str(sticky_existing.get("organizer") or "")
        if not location:
            location = str(sticky_existing.get("location") or "")
        if not meeting_url:
            meeting_url = str(sticky_existing.get("meeting_url") or "")
        attendees = merge_people(attendees, existing_attendees)
        for t in existing_topics:
            key = str(t).strip().lower()
            if key and key not in topic_seen:
                topic_seen.add(key)
                topics.append(str(t).strip())
        topics = topics[:12]

    if not title:
        raise PlannerError("event capture needs a title", status_code=422)

    source = (
        "outlook"
        if any(h in source_url.lower() for h in ("outlook.", "teams.microsoft", "teams.cloud"))
        else "capture"
    )
    # Metadata only — snippet body is appended below so images are not dropped
    # and re-saves do not wipe prior clips.
    event = upsert_event(
        cfg,
        title=title,
        start=start,
        end=end,
        source=source,
        source_url=source_url,
        attendees=attendees,
        topics=topics,
        organizer=organizer,
        location=location,
        meeting_url=meeting_url,
        body="",
        event_id=event_id_hint,
        store=store,
    )
    # If the client sent a sticky path that differs (rare), prefer fingerprint hit.
    path = str(event.get("path") or path_hint)
    written, skipped = _append_snippets_to_event(
        cfg, path=path, snippets=snippets, store=store
    )
    return {
        "mode": "event",
        "path": path,
        "event_id": event["id"],
        "written": written,
        "skipped": skipped,
        "title": event["title"],
    }


def link_event(
    cfg: Any,
    event_id: str,
    *,
    session_id: Optional[str],
    store: Any = None,
) -> dict[str, Any]:
    """Set or clear ``session_id`` on an event note."""
    if store is None:
        store = vault_store_from_config(cfg)
    ev = find_event_by_id(cfg, event_id, store=store)
    path = ev["path"]
    try:
        text = store.read(path)
    except VaultNotFound as exc:
        raise PlannerError(f"event not found: {event_id}", status_code=404) from exc
    meta, body = parse_frontmatter(text)
    sid = (session_id or "").strip()
    if sid:
        meta["session_id"] = sid
    else:
        meta.pop("session_id", None)
    updated = dump_frontmatter(meta, body)
    try:
        store.write(path, updated)
    except VaultWriteDenied as exc:
        raise PlannerError(str(exc), status_code=403) from exc
    publish_vault_event([path], source="planner")
    return parse_event_note(updated, path=path)


def delete_event(
    cfg: Any,
    event_id: str,
    *,
    store: Any = None,
) -> dict[str, Any]:
    """Permanently delete one planner event note from the vault."""
    if store is None:
        store = vault_store_from_config(cfg)
    ev = find_event_by_id(cfg, event_id, store=store)
    path = str(ev.get("path") or "")
    if not path:
        raise PlannerError(f"event not found: {event_id}", status_code=404)
    try:
        store.delete(path)
    except VaultNotFound as exc:
        raise PlannerError(f"event not found: {event_id}", status_code=404) from exc
    except VaultWriteDenied as exc:
        raise PlannerError(str(exc), status_code=403) from exc
    except Exception as exc:  # noqa: BLE001
        raise PlannerError(f"delete failed: {exc}", status_code=500) from exc
    publish_vault_event([path], source="planner")
    return {
        "id": ev.get("id") or event_id,
        "path": path,
        "deleted": True,
        "title": ev.get("title") or "",
    }


def list_events(
    cfg: Any,
    *,
    date_from: str = "",
    date_to: str = "",
    store: Any = None,
) -> list[dict[str, Any]]:
    if store is None:
        store = vault_store_from_config(cfg)
    start_d = day_from_local(date_from) if date_from else ""
    end_d = day_from_local(date_to) if date_to else ""
    keys = _list_event_keys(cfg, store)
    # Single-day: only read notes whose filename starts with YYYY-MM-DD-
    if start_d and start_d == end_d:
        prefix = f"{start_d}-"
        keys = [k for k in keys if k.rsplit("/", 1)[-1].startswith(prefix)]
    out: list[dict[str, Any]] = []
    for key in keys:
        try:
            text = store.read(key)
            ev = parse_event_note(text, path=key)
        except Exception as exc:  # noqa: BLE001
            logger.debug("skip event %s: %s", key, exc)
            continue
        day = ev.get("day") or day_from_local(ev.get("start") or "")
        if start_d and day and day < start_d:
            continue
        if end_d and day and day > end_d:
            continue
        if (start_d or end_d) and not day:
            continue
        out.append(ev)
    out.sort(key=lambda e: (e.get("start") or "", e.get("title") or ""))
    return out


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 2}


def title_similarity(a: str, b: str) -> float:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def _transcript_path_db(cfg: Any, session_id: str) -> Optional[str]:
    try:
        from pawn_core.vault_db import get_vault_note  # noqa: PLC0415

        row = get_vault_note(cfg.db_dsn, session_id)
        if row is not None and row.key:
            return str(row.key)
    except Exception as exc:  # noqa: BLE001
        logger.debug("vault_notes lookup %s failed: %s", session_id, exc)
    return None


def _candidate_to_session(cfg: Any, cand: Any, *, day: str) -> dict[str, Any]:
    sid = str(cand.session_id)
    return {
        "session_id": sid,
        "title": cand.title or sid,
        "created_at": cand.created_at.isoformat() if getattr(cand, "created_at", None) else None,
        "updated_at": cand.updated_at.isoformat() if getattr(cand, "updated_at", None) else None,
        "segments": cand.segments,
        "duration_seconds": cand.duration_seconds,
        "summary": cand.summary,
        "transcript_path": _transcript_path_db(cfg, sid),
        "day": day,
    }


def _sessions_for_day(cfg: Any, day_key: str) -> list[dict[str, Any]]:
    """Lightweight sessions whose created/updated day matches *day_key*.

    Avoids ``list_sessions_page`` enrichment and per-transcript S3 reads that
    made ``GET /v1/planner/day`` hang for large vaults.
    """
    from pawn_agent.tools.list_sessions import list_session_candidates_impl  # noqa: PLC0415

    try:
        candidates = list_session_candidates_impl(cfg, name_filter="", limit=300)
    except Exception as exc:  # noqa: BLE001
        logger.warning("planner sessions list failed: %s", exc)
        return []
    out: list[dict[str, Any]] = []
    for cand in candidates:
        created = day_from_local(
            cand.created_at.isoformat() if getattr(cand, "created_at", None) else ""
        )
        updated = day_from_local(
            cand.updated_at.isoformat() if getattr(cand, "updated_at", None) else ""
        )
        if created != day_key and updated != day_key:
            continue
        out.append(_candidate_to_session(cfg, cand, day=day_key))
    return out


def _session_by_id(cfg: Any, session_id: str, *, day: str) -> Optional[dict[str, Any]]:
    sid = (session_id or "").strip()
    if not sid:
        return None
    from pawn_agent.tools.list_sessions import list_session_candidates_impl  # noqa: PLC0415

    try:
        hits = list_session_candidates_impl(cfg, name_filter=sid, limit=5)
    except Exception:  # noqa: BLE001
        return {
            "session_id": sid,
            "title": sid,
            "transcript_path": _transcript_path_db(cfg, sid),
            "day": day,
        }
    for cand in hits:
        if str(cand.session_id) == sid:
            return _candidate_to_session(cfg, cand, day=day)
    return {
        "session_id": sid,
        "title": sid,
        "transcript_path": _transcript_path_db(cfg, sid),
        "day": day,
    }


def _suggest_links(
    events: list[dict[str, Any]], sessions: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    suggestions: list[dict[str, Any]] = []
    unlinked = [e for e in events if not e.get("session_id")]
    for ev in unlinked:
        best: Optional[dict[str, Any]] = None
        best_score = 0.0
        for sess in sessions:
            score = title_similarity(ev.get("title") or "", sess.get("title") or "")
            if score < 0.25:
                continue
            if score > best_score:
                best_score = score
                best = sess
        if best is not None:
            suggestions.append(
                {
                    "event_id": ev.get("id"),
                    "event_path": ev.get("path"),
                    "session_id": best.get("session_id"),
                    "session_title": best.get("title"),
                    "score": round(best_score, 3),
                }
            )
    return suggestions


def _items_for_session(cfg: Any, session_id: str) -> list[dict[str, Any]]:
    if not session_id:
        return []
    try:
        from pawn_agent.core.coworker import db as itemdb  # noqa: PLC0415

        return itemdb.list_items(
            cfg.db_dsn,
            statuses=list(itemdb.OPEN_STATUSES),
            source_ref=session_id,
            limit=50,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("items for session %s failed: %s", session_id, exc)
        return []


def _open_items_for_day(cfg: Any, day: str) -> list[dict[str, Any]]:
    """Open items created on *day* only (never dump undated into every day)."""
    try:
        from pawn_agent.core.coworker import db as itemdb  # noqa: PLC0415

        rows = itemdb.list_items(
            cfg.db_dsn,
            statuses=list(itemdb.OPEN_STATUSES),
            limit=200,
        )
    except Exception as exc:  # noqa: BLE001
        logger.debug("open items list failed: %s", exc)
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        created = day_from_local(str(row.get("created_at") or ""))
        if created == day:
            out.append(row)
    return out


def build_day(
    cfg: Any,
    day: str,
    *,
    store: Any = None,
) -> dict[str, Any]:
    """Merged timeline for one local calendar day."""
    if store is None:
        store = vault_store_from_config(cfg)
    day_key = day_from_local(day) or _today(cfg)

    events = list(list_events(cfg, date_from=day_key, date_to=day_key, store=store))
    sessions = _sessions_for_day(cfg, day_key)
    by_id = {str(s.get("session_id") or ""): s for s in sessions}

    # Pull linked sessions that fall outside the created_at day window.
    for ev in events:
        sid = str(ev.get("session_id") or "").strip()
        if sid and sid not in by_id:
            linked = _session_by_id(cfg, sid, day=day_key)
            if linked:
                sessions.append(linked)
                by_id[sid] = linked

    blocks: list[dict[str, Any]] = []
    used_sessions: set[str] = set()
    for ev in events:
        sid = str(ev.get("session_id") or "").strip()
        sess = by_id.get(sid) if sid else None
        if sid:
            used_sessions.add(sid)
        blocks.append(
            {
                "kind": "event",
                "sort_key": ev.get("start") or day_key,
                "event": ev,
                "session": sess,
                "items": _items_for_session(cfg, sid) if sid else [],
            }
        )

    for sess in sessions:
        sid = str(sess.get("session_id") or "")
        if not sid or sid in used_sessions:
            continue
        blocks.append(
            {
                "kind": "session",
                "sort_key": sess.get("created_at") or day_key,
                "event": None,
                "session": sess,
                "items": _items_for_session(cfg, sid),
            }
        )

    blocks.sort(key=lambda b: str(b.get("sort_key") or ""))

    meeting_item_ids = {
        str(it.get("id") or "")
        for b in blocks
        for it in (b.get("items") or [])
    }
    meeting_refs: set[str] = set()
    for b in blocks:
        sess = b.get("session")
        if isinstance(sess, dict):
            sid = str(sess.get("session_id") or "").strip()
            if sid:
                meeting_refs.add(sid)
    other_items = [
        it
        for it in _open_items_for_day(cfg, day_key)
        if str(it.get("id") or "") not in meeting_item_ids
        and str(it.get("source_ref") or "") not in meeting_refs
    ]

    return {
        "date": day_key,
        "events": events,
        "sessions": sessions,
        "blocks": blocks,
        "other_items": other_items,
        "suggestions": _suggest_links(events, sessions),
    }


def parse_event_slash_arg(arg: str) -> tuple[str, str]:
    """Split ``/event Title [when]`` — trailing ISO-ish token becomes start."""
    raw = (arg or "").strip()
    if not raw:
        return "", ""
    parts = raw.rsplit(None, 1)
    if len(parts) == 2 and _LOCAL_DT_RE.match(parts[1].replace(" ", "T")):
        return parts[0].strip(), normalize_local_dt(parts[1])
    # Also accept ``Title on YYYY-MM-DD``
    lowered = raw.lower()
    if " on " in lowered:
        idx = lowered.rfind(" on ")
        when = raw[idx + 4 :].strip()
        if _LOCAL_DT_RE.match(when.replace(" ", "T")):
            return raw[:idx].strip(), normalize_local_dt(when)
    return raw, ""
