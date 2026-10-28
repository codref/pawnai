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


def day_from_local(value: str) -> str:
    """Extract ``YYYY-MM-DD`` from a local start/end string."""
    raw = (value or "").strip()
    match = _LOCAL_DT_RE.match(raw)
    if match:
        return match.group(1)
    if len(raw) >= 10 and raw[4] == "-" and raw[7] == "-":
        return raw[:10]
    return ""


def normalize_local_dt(value: str) -> str:
    """Normalize to ``YYYY-MM-DD`` or ``YYYY-MM-DDTHH:MM`` (no TZ)."""
    raw = (value or "").strip().replace(" ", "T")
    match = _LOCAL_DT_RE.match(raw)
    if not match:
        return raw
    day, hm = match.group(1), match.group(2)
    if hm:
        return f"{day}T{hm}"
    return day


def event_fingerprint(
    *,
    source_url: str = "",
    title: str = "",
    start: str = "",
) -> str:
    url = (source_url or "").strip()
    if url:
        return hashlib.sha256(f"url:{url}".encode("utf-8")).hexdigest()[:24]
    key = f"title:{(title or '').strip().lower()}|{normalize_local_dt(start)}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]


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
    session_id: str = "",
    fingerprint: str = "",
    prep_job_id: str = "",
    prep_at: str = "",
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
        or event_fingerprint(source_url=source_url, title=heading, start=start),
    }
    if source_url.strip():
        meta["source_url"] = source_url.strip()
    if attendees:
        meta["attendees"] = [a for a in attendees if a]
    if topics:
        meta["topics"] = [t for t in topics if t]
    if session_id.strip():
        meta["session_id"] = session_id.strip()
    if prep_job_id.strip():
        meta["prep_job_id"] = prep_job_id.strip()
    if prep_at.strip():
        meta["prep_at"] = prep_at.strip()

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
    return {
        "id": _meta_str(meta.get("id")),
        "title": title,
        "start": normalize_local_dt(_meta_str(meta.get("start"))),
        "end": normalize_local_dt(_meta_str(meta.get("end"))),
        "source": _meta_str(meta.get("source")) or "manual",
        "source_url": _meta_str(meta.get("source_url")),
        "attendees": _meta_list(meta.get("attendees")),
        "topics": _meta_list(meta.get("topics")),
        "session_id": _meta_str(meta.get("session_id")),
        "fingerprint": _meta_str(meta.get("fingerprint")),
        "prep_job_id": _meta_str(meta.get("prep_job_id")),
        "prep_at": _meta_str(meta.get("prep_at")),
        "path": path,
        "body": rest,
        "prep": prep,
        "day": day_from_local(_meta_str(meta.get("start"))) or "",
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
    fp = event_fingerprint(source_url=source_url, title=heading, start=start_n)
    existing = find_event_by_fingerprint(cfg, fp, store=store)
    if existing is None and event_id.strip():
        try:
            existing = find_event_by_id(cfg, event_id, store=store)
        except PlannerError:
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
        except VaultNotFound:
            prep = existing.get("prep") or ""
            prep_job_id = existing.get("prep_job_id") or ""
            prep_at = existing.get("prep_at") or ""
            linked = session_id.strip() or existing.get("session_id") or ""
        text = render_event_note(
            event_id=eid,
            title=heading,
            start=start_n,
            end=end_n or existing.get("end") or "",
            source=source or existing.get("source") or "manual",
            source_url=source_url or existing.get("source_url") or "",
            attendees=attendees if attendees is not None else existing.get("attendees"),
            topics=topics if topics is not None else existing.get("topics"),
            session_id=linked,
            fingerprint=fp,
            prep_job_id=prep_job_id,
            prep_at=prep_at,
            body=body if body.strip() else existing.get("body") or "",
            prep=prep,
        )
        try:
            store.write(path, text)
        except VaultWriteDenied as exc:
            raise PlannerError(str(exc), status_code=403) from exc
        publish_vault_event([path], source="planner")
        return parse_event_note(text, path=path)

    short_id = uuid4().hex[:8]
    eid = (event_id or "").strip() or f"evt_{short_id}"
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
    path_hint = normalize_vault_key(str(target.get("path") or "").strip())
    event_id_hint = str(target.get("id") or target.get("event_id") or "").strip()

    for raw in snippets:
        if not source_url:
            source_url = str(raw.get("source_url") or "").strip()
        if not title and str(raw.get("kind") or "text") == "text":
            line = str(raw.get("text") or "").strip().splitlines()
            if line:
                title = line[0][:80]

    if path_hint:
        # Sticky path from the extension — append to that note, keep its identity.
        try:
            existing = parse_event_note(store.read(path_hint), path=path_hint)
            event_id_hint = event_id_hint or str(existing.get("id") or "")
            if not title:
                title = str(existing.get("title") or "") or "event"
            if not start:
                start = str(existing.get("start") or "")
            if not end:
                end = str(existing.get("end") or "")
            if not source_url:
                source_url = str(existing.get("source_url") or "")
            if attendees is None or not attendees:
                attendees = list(existing.get("attendees") or [])
        except Exception:  # noqa: BLE001
            pass
    if not title:
        raise PlannerError("event capture needs a title", status_code=422)

    source = "outlook" if "outlook" in source_url.lower() else "capture"
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
