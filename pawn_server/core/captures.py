"""Browser snippet capture — append ordered text/image blocks to a vault note.

Used by the Pawn browser extension (no chat). Capture notes live under
``{agent_root}/Captures/``. Images go to ``{agent_root}/Captures/assets/``.
Research mode writes one inbox atom per snippet under ``capture.inbox_dir``.
Attaching to a diarization session splices into that transcript's Annotations
section when a vault mapping exists.
"""

from __future__ import annotations

import base64
import binascii
import logging
import re
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from pawn_agent.core.vision import MAX_IMAGE_BYTES
from pawn_agent.tools.ideas_impl import idea_filename_title
from pawn_core.vault import (
    VaultNotFound,
    VaultWriteDenied,
    dump_frontmatter,
    normalize_vault_key,
    parse_frontmatter,
)
from pawn_core.vault_config import vault_store_from_config
from pawn_core.vault_db import get_vault_note
from pawn_server.core import capture_config as capcfg
from pawn_server.core.vault_events import publish_vault_event

logger = logging.getLogger(__name__)

CAPTURES_SUBDIR = "Captures"
ASSETS_SUBDIR = "Captures/assets"

_SNIPPET_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")
_ANNOTATIONS_RE = re.compile(
    r"(?is)^##\s*Annotations\s*\n+(.*?)(?=^##\s|\Z)",
    re.MULTILINE,
)
_CAPTURES_SECTION_RE = re.compile(
    r"(?is)^##\s*Captures\s*\n+(.*?)(?=^##\s|\Z)",
    re.MULTILINE,
)
TARGET_KINDS = frozenset({"new", "capture", "note", "session", "research", "event"})
SNIPPET_KINDS = frozenset({"text", "image"})
CAPTURE_STATUSES = frozenset({"inbox", "proposed", "filed", "ignored"})
# Overlay registration only for capture-family notes (not session/note/event).
OVERLAY_TARGET_KINDS = frozenset({"new", "capture", "research"})
_ANCHOR_SCAN_LIMIT = 200


def normalize_anchor(raw: Any, *, snippet_id: str) -> dict[str, Any] | None:
    """Normalize a client anchor dict; None when missing or invalid."""
    if not isinstance(raw, dict):
        return None
    site = str(raw.get("site") or "").strip()
    if not site:
        return None
    sid = (snippet_id or "").strip()
    if not sid or not _SNIPPET_ID_RE.match(sid):
        return None
    page_key = str(raw.get("page_key") or "").strip()
    message_id = str(raw.get("message_id") or "").strip()
    deep_link = str(raw.get("deep_link") or "").strip()
    fallback_raw = raw.get("fallback")
    fallback: dict[str, str] = {}
    if isinstance(fallback_raw, dict):
        for key in ("author", "text_quote", "approx_time"):
            val = str(fallback_raw.get(key) or "").strip()
            if val:
                fallback[key] = val
    out: dict[str, Any] = {"snippet_id": sid, "site": site}
    if page_key:
        out["page_key"] = page_key
    if message_id:
        out["message_id"] = message_id
    if deep_link:
        out["deep_link"] = deep_link
    if fallback:
        out["fallback"] = fallback
    # Need at least a page/message id or a soft-match fallback.
    if not page_key and not message_id and not fallback:
        return None
    return out


def _anchor_dedupe_key(anchor: dict[str, Any]) -> str:
    sid = str(anchor.get("snippet_id") or "").strip()
    if sid:
        return f"snip:{sid}"
    mid = str(anchor.get("message_id") or "").strip()
    pk = str(anchor.get("page_key") or "").strip()
    if mid and pk:
        return f"msg:{pk}:{mid}"
    if mid:
        return f"msg:{mid}"
    return f"row:{id(anchor)}"


def merge_anchors_list(
    existing: list[Any] | None,
    incoming: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge anchor rows; prefer newer fields; dedupe by snippet_id then message_id."""
    merged: list[dict[str, Any]] = []
    index: dict[str, int] = {}
    for row in list(existing or []) + list(incoming or []):
        if not isinstance(row, dict):
            continue
        sid = str(row.get("snippet_id") or "").strip()
        norm = normalize_anchor(row, snippet_id=sid) if sid else None
        if norm is None and sid and str(row.get("site") or "").strip():
            # Already-normalized vault rows may lack fallback/ids combined.
            site = str(row.get("site") or "").strip()
            if not site:
                continue
            norm = {"snippet_id": sid, "site": site}
            for key in ("page_key", "message_id", "deep_link"):
                val = str(row.get(key) or "").strip()
                if val:
                    norm[key] = val
            fb = row.get("fallback")
            if isinstance(fb, dict) and fb:
                norm["fallback"] = {
                    k: str(v).strip()
                    for k, v in fb.items()
                    if str(v).strip() and k in {"author", "text_quote", "approx_time"}
                }
        if not norm:
            continue
        key = _anchor_dedupe_key(norm)
        if key in index:
            prev = merged[index[key]]
            combined = {**prev, **norm}
            if isinstance(prev.get("fallback"), dict) or isinstance(norm.get("fallback"), dict):
                combined["fallback"] = {
                    **(prev.get("fallback") if isinstance(prev.get("fallback"), dict) else {}),
                    **(norm.get("fallback") if isinstance(norm.get("fallback"), dict) else {}),
                }
            merged[index[key]] = combined
        else:
            index[key] = len(merged)
            merged.append(norm)
    return merged


def apply_anchors_to_markdown(
    text: str,
    anchors: list[dict[str, Any]],
    *,
    prefer_deep_link_source: bool = True,
) -> str:
    """Write/merge ``anchors`` into note frontmatter; optionally set source_url."""
    if not anchors:
        return text
    meta, body = parse_frontmatter(text or "")
    meta["anchors"] = merge_anchors_list(meta.get("anchors"), anchors)
    if prefer_deep_link_source:
        existing_src = str(meta.get("source_url") or "").strip()
        if not existing_src:
            for row in meta["anchors"]:
                deep = str(row.get("deep_link") or "").strip()
                if deep:
                    meta["source_url"] = deep
                    break
    return dump_frontmatter(meta, body)


def snippet_source_url(raw: dict[str, Any], fallback: str = "") -> str:
    """Prefer snippet source_url, then anchor deep_link, then target fallback."""
    direct = str(raw.get("source_url") or "").strip()
    if direct:
        return direct
    anchor = raw.get("anchor")
    if isinstance(anchor, dict):
        deep = str(anchor.get("deep_link") or "").strip()
        if deep:
            return deep
    return (fallback or "").strip()


def _title_from_capture_text(text: str, key: str) -> str:
    title = _title_from_path(key)
    for line in (text or "").splitlines():
        if line.startswith("# "):
            return line[2:].strip() or title
    return title


class CaptureError(Exception):
    """Client-facing capture failure with an HTTP status."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def _agent_root(cfg: Any) -> str:
    root = getattr(getattr(cfg, "vault", None), "agent_root", None) or "Pawn"
    return str(root).strip().strip("/") or "Pawn"


def captures_dir(cfg: Any) -> str:
    return f"{_agent_root(cfg)}/{CAPTURES_SUBDIR}"


def assets_dir(cfg: Any) -> str:
    return f"{_agent_root(cfg)}/{ASSETS_SUBDIR}"


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _today(cfg: Any) -> str:
    tzname = (getattr(getattr(cfg, "coworker", None), "timezone", "") or "UTC").strip() or "UTC"
    try:
        from zoneinfo import ZoneInfo, ZoneInfoNotFoundError  # noqa: PLC0415

        try:
            return datetime.now(ZoneInfo(tzname)).date().isoformat()
        except ZoneInfoNotFoundError:
            pass
    except Exception:
        pass
    return datetime.now(timezone.utc).date().isoformat()


def _snippet_marker(snippet_id: str) -> str:
    return f"<!-- pawn-snippet:{snippet_id} -->"


def _snippet_end(snippet_id: str) -> str:
    return f"<!-- /pawn-snippet:{snippet_id} -->"


def note_has_snippet(body: str, snippet_id: str) -> bool:
    return _snippet_marker(snippet_id) in (body or "")


def strip_snippet(body: str, snippet_id: str) -> str:
    """Remove one snippet block from *body* (idempotent)."""
    if not body or not _SNIPPET_ID_RE.match(snippet_id):
        return body or ""
    pattern = re.compile(
        rf"<!--\s*pawn-snippet:{re.escape(snippet_id)}\s*-->"
        rf".*?"
        rf"<!--\s*/pawn-snippet:{re.escape(snippet_id)}\s*-->\n?",
        re.DOTALL,
    )
    return pattern.sub("", body)


def _title_from_path(path: str) -> str:
    stem = normalize_vault_key(path).removesuffix(".md").split("/")[-1]
    # Strip leading date "YYYY-MM-DD "
    if len(stem) > 11 and stem[4] == "-" and stem[7] == "-" and stem[10] == " ":
        return stem[11:] or stem
    return stem


def _new_capture_path(cfg: Any, title: str, store: Any) -> str:
    day = _today(cfg)
    stem = idea_filename_title(title or "capture")
    base = f"{captures_dir(cfg)}/{day} {stem}.md"
    if not store.exists(base):
        return base
    for _ in range(8):
        candidate = f"{captures_dir(cfg)}/{day} {stem}-{uuid4().hex[:6]}.md"
        if not store.exists(candidate):
            return candidate
    return f"{captures_dir(cfg)}/{day} {stem}-{uuid4().hex[:8]}.md"


def _render_new_capture(
    *,
    title: str,
    source_url: str = "",
    session_id: str = "",
) -> str:
    heading = idea_filename_title(title or "capture")
    meta: dict[str, Any] = {
        "pawn": "capture",
        "tags": ["pawn/capture"],
        "created": _now_iso(),
    }
    if source_url.strip():
        meta["source_url"] = source_url.strip()
    if session_id.strip():
        meta["session_id"] = session_id.strip()
    lines = [f"# {heading}", ""]
    if source_url.strip():
        lines.extend([f"Source: {source_url.strip()}", ""])
    lines.append("## Snippets")
    lines.append("")
    return dump_frontmatter(meta, "\n".join(lines))


def render_snippet_block(
    *,
    snippet_id: str,
    kind: str,
    text: str = "",
    image_key: str = "",
    source_url: str = "",
    captured_at: str = "",
) -> str:
    when = (captured_at or "").strip() or _now_iso()
    lines = [
        _snippet_marker(snippet_id),
        f"### Snippet · {when}",
        "",
    ]
    if source_url.strip():
        lines.extend([f"Source: {source_url.strip()}", ""])
    if kind == "image" and image_key:
        lines.extend([f"![[{image_key}]]", ""])
    elif text.strip():
        lines.extend([text.strip(), ""])
    else:
        lines.extend(["_(empty)_", ""])
    lines.append(_snippet_end(snippet_id))
    lines.append("")
    return "\n".join(lines)


def replace_annotations(markdown: str, annotations_body: str) -> str:
    """Swap the Annotations section body, or insert the section if missing."""
    body = annotations_body.strip()
    if body and not body.endswith("\n"):
        body += "\n"
    if not body:
        body = "_(Add notes and tags here.)_\n"
    match = _ANNOTATIONS_RE.search(markdown or "")
    if match:
        start, end = match.span(1)
        return (markdown or "")[:start] + body + (markdown or "")[end:]
    section = f"\n## Annotations\n{body}"
    for heading in ("## Screenshots", "## Transcript"):
        idx = (markdown or "").find(heading)
        if idx >= 0:
            return (markdown or "")[:idx].rstrip() + "\n" + section + "\n" + (markdown or "")[idx:]
    text = markdown or ""
    if text and not text.endswith("\n"):
        text += "\n"
    return text + section


def extract_annotations(markdown: str) -> str:
    """Return the Annotations section body (empty string when missing)."""
    match = _ANNOTATIONS_RE.search(markdown or "")
    if not match:
        return ""
    return match.group(1)


def _decode_image(data_base64: str) -> bytes:
    raw = (data_base64 or "").strip()
    if not raw:
        raise CaptureError("image snippet missing data_base64", status_code=422)
    if "," in raw and raw.lower().startswith("data:"):
        raw = raw.split(",", 1)[1]
    try:
        data = base64.b64decode(raw, validate=False)
    except (binascii.Error, ValueError) as exc:
        raise CaptureError(f"invalid image base64: {exc}", status_code=422) from exc
    if not data:
        raise CaptureError("empty image data", status_code=422)
    if len(data) > MAX_IMAGE_BYTES:
        raise CaptureError(
            f"image exceeds {MAX_IMAGE_BYTES} bytes",
            status_code=422,
        )
    return data


def _image_suffix(media_type: str) -> str:
    mt = (media_type or "").strip().lower()
    if mt in {"image/jpeg", "image/jpg"}:
        return ".jpg"
    if mt == "image/gif":
        return ".gif"
    if mt == "image/webp":
        return ".webp"
    return ".png"


def _content_type(suffix: str) -> str:
    return {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(suffix, "image/png")


def list_captures(cfg: Any, *, limit: int = 40, store: Any = None) -> list[dict[str, Any]]:
    """Recent capture notes under ``{agent_root}/Captures/``."""
    if store is None:
        store = vault_store_from_config(cfg)
    folder = captures_dir(cfg)
    keys = [k for k in store.list(folder) if "/assets/" not in k.replace("\\", "/")]
    keys.sort(reverse=True)
    out: list[dict[str, Any]] = []
    for key in keys[: max(1, min(int(limit), 200))]:
        title = _title_from_path(key)
        session_id = None
        source_url = ""
        anchors: list[dict[str, Any]] = []
        try:
            text = store.read(key)
            meta, _ = parse_frontmatter(text)
            if isinstance(meta.get("session_id"), str) and meta["session_id"].strip():
                session_id = meta["session_id"].strip()
            source_url = str(meta.get("source_url") or "").strip()
            raw_anchors = meta.get("anchors")
            if isinstance(raw_anchors, list):
                anchors = merge_anchors_list(None, [a for a in raw_anchors if isinstance(a, dict)])
            title = _title_from_capture_text(text, key)
        except Exception:
            pass
        row: dict[str, Any] = {"path": key, "title": title, "session_id": session_id}
        if source_url:
            row["source_url"] = source_url
        if anchors:
            row["anchors"] = anchors
        out.append(row)
    return out


def _iter_capture_note_keys(cfg: Any, store: Any) -> list[str]:
    """Captures folder + research inbox (assets excluded), newest-first."""
    keys: list[str] = []
    seen: set[str] = set()
    folders = [captures_dir(cfg)]
    inbox = capcfg.inbox_dir(cfg)
    if inbox and inbox not in folders:
        folders.append(inbox)
    for folder in folders:
        try:
            listed = store.list(folder)
        except Exception as exc:
            logger.debug("list capture keys under %s failed: %s", folder, exc)
            continue
        for key in listed:
            norm = normalize_vault_key(key)
            if "/assets/" in norm.replace("\\", "/"):
                continue
            if not norm.lower().endswith(".md"):
                continue
            if norm in seen:
                continue
            seen.add(norm)
            keys.append(norm)
    keys.sort(reverse=True)
    return keys


def list_anchors(
    cfg: Any,
    *,
    page_key: str = "",
    site: str = "",
    limit: int = _ANCHOR_SCAN_LIMIT,
    store: Any = None,
) -> list[dict[str, Any]]:
    """Reverse-lookup overlay anchors by page_key and/or site.

    ``page_key`` filters to one conversation when set. When omitted, ``site``
    is required and recent anchors for that site are returned (soft-match).
    """
    pk = (page_key or "").strip()
    site_filter = (site or "").strip().lower()
    if not pk and not site_filter:
        raise CaptureError("page_key or site is required", status_code=422)
    if store is None:
        store = vault_store_from_config(cfg)
    scan_cap = max(1, min(int(limit), _ANCHOR_SCAN_LIMIT))
    out: list[dict[str, Any]] = []
    for key in _iter_capture_note_keys(cfg, store)[:scan_cap]:
        try:
            text = store.read(key)
            meta, _ = parse_frontmatter(text)
        except Exception:
            continue
        raw_anchors = meta.get("anchors")
        if not isinstance(raw_anchors, list) or not raw_anchors:
            continue
        title = _title_from_capture_text(text, key)
        for row in merge_anchors_list(None, [a for a in raw_anchors if isinstance(a, dict)]):
            row_site = str(row.get("site") or "").strip().lower()
            row_pk = str(row.get("page_key") or "").strip()
            if site_filter and row_site != site_filter:
                continue
            # Exact page_key match, or keep empty page_key rows for soft-match.
            if pk and row_pk and row_pk != pk:
                continue
            item = {
                "path": key,
                "title": title,
                "snippet_id": row.get("snippet_id"),
                "site": row.get("site"),
                "page_key": row.get("page_key"),
            }
            for field in ("message_id", "deep_link", "fallback"):
                if row.get(field):
                    item[field] = row[field]
            out.append(item)
    return out


def list_sessions(
    cfg: Any,
    *,
    limit: int = 20,
    q: str = "",
    needs_speakers: bool = False,
) -> list[dict[str, Any]]:
    """Recent diarization sessions with transcript vault path when mapped."""
    from pawn_server.core.sessions_api import list_sessions as list_sessions_enriched

    return list_sessions_enriched(cfg, limit=limit, q=q, needs_speakers=needs_speakers)


def _resolve_target(
    cfg: Any,
    store: Any,
    *,
    kind: str,
    path: str = "",
    session_id: str = "",
    title: str = "",
    source_url: str = "",
) -> tuple[str, str]:
    """Return ``(vault_key, mode)`` where mode is ``append`` or ``annotations``."""
    kind = (kind or "").strip().lower()
    if kind not in TARGET_KINDS:
        raise CaptureError(
            f"unknown target kind {kind!r} " "(use new, capture, note, session, research, event)",
            status_code=422,
        )
    if kind == "research":
        raise CaptureError("research targets use save_research_snippets", status_code=500)
    if kind == "event":
        raise CaptureError("event targets use upsert_event_from_capture", status_code=500)
    if kind == "new":
        key = normalize_vault_key(path) if path.strip() else ""
        if key and not key.lower().endswith(".md"):
            key = f"{key}.md"
        if key:
            try:
                store.read(key)
                return key, "append"
            except VaultNotFound:
                body = _render_new_capture(
                    title=title or _title_from_path(key) or "capture",
                    source_url=source_url,
                    session_id=session_id,
                )
                try:
                    store.write(key, body)
                except VaultWriteDenied as exc:
                    raise CaptureError(str(exc), status_code=403) from exc
                return key, "append"
            except Exception:
                pass
        key = _new_capture_path(cfg, title or "capture", store)
        body = _render_new_capture(
            title=title or "capture",
            source_url=source_url,
            session_id=session_id,
        )
        try:
            store.write(key, body)
        except VaultWriteDenied as exc:
            raise CaptureError(str(exc), status_code=403) from exc
        return key, "append"

    if kind in {"capture", "note"}:
        key = normalize_vault_key(path)
        if not key:
            raise CaptureError("path is required", status_code=422)
        if not key.lower().endswith(".md"):
            key = f"{key}.md"
        return key, "append"

    # session
    sid = (session_id or "").strip()
    if not sid:
        raise CaptureError("session_id is required", status_code=422)
    row = get_vault_note(cfg.db_dsn, sid)
    if row is not None and row.key:
        key = normalize_vault_key(row.key)
        try:
            store.read(key)
            return key, "annotations"
        except VaultNotFound:
            pass
        except Exception as exc:
            logger.debug("transcript %s unreadable: %s", key, exc)
    # No transcript yet — create / reuse a capture note tagged with session_id
    key = normalize_vault_key(path) if path.strip() else ""
    if key:
        if not key.lower().endswith(".md"):
            key = f"{key}.md"
        try:
            store.read(key)
            return key, "append"
        except VaultNotFound:
            pass
    key = _new_capture_path(cfg, title or sid, store)
    body = _render_new_capture(
        title=title or sid,
        source_url=source_url,
        session_id=sid,
    )
    try:
        store.write(key, body)
    except VaultWriteDenied as exc:
        raise CaptureError(str(exc), status_code=403) from exc
    return key, "append"


def _append_blocks(existing: str, blocks: list[str], *, mode: str) -> str:
    if mode == "annotations":
        ann = extract_annotations(existing)
        # Drop the default stub when we start adding real snippets
        if ann.strip() in {"_(Add notes and tags here.)_", ""}:
            ann = ""
        for block in blocks:
            # Idempotency checked before calling; still skip if somehow present
            marker = block.split("\n", 1)[0]
            if marker and marker in ann:
                continue
            if ann and not ann.endswith("\n"):
                ann += "\n"
            if ann and not ann.endswith("\n\n"):
                ann += "\n"
            ann += block if block.endswith("\n") else block + "\n"
        return replace_annotations(existing, ann)

    text = existing or ""
    meta, _ = parse_frontmatter(text)
    if "## Snippets" not in text and meta.get("pawn") == "capture":
        if text and not text.endswith("\n"):
            text += "\n"
        text += "\n## Snippets\n\n"
    for block in blocks:
        marker = block.split("\n", 1)[0]
        if marker and marker in text:
            continue
        if text and not text.endswith("\n"):
            text += "\n"
        if text and not text.endswith("\n\n"):
            text += "\n"
        text += block if block.endswith("\n") else block + "\n"
    return text


def _research_atom_path(
    cfg: Any,
    *,
    title: str,
    snippet_id: str,
    store: Any,
) -> str:
    day = _today(cfg)
    stem = idea_filename_title(title or "capture")
    folder = capcfg.inbox_dir(cfg)
    base = f"{folder}/{day} {stem}-{snippet_id}.md"
    if not store.exists(base):
        return base
    # Same snippet id already has a note — reuse for idempotency.
    return base


def _render_research_atom(
    *,
    title: str,
    snippet_id: str,
    kind: str,
    source_url: str = "",
    collection: str = "",
    hint: str = "",
    captured_at: str = "",
    image_key: str = "",
    text: str = "",
    anchors: list[dict[str, Any]] | None = None,
) -> str:
    heading = idea_filename_title(title or "capture")
    meta: dict[str, Any] = {
        "pawn": "capture",
        "status": "inbox",
        "tags": ["pawn/capture"],
        "snippet_id": snippet_id,
        "kind": kind,
        "created": _now_iso(),
        "collection": (collection or "").strip(),
        "hint": (hint or "").strip(),
        "entity": "",
        "type": "",
        "caption": "",
        "proposed_tags": [],
        "enriched_at": "",
    }
    if source_url.strip():
        meta["source_url"] = source_url.strip()
    if captured_at.strip():
        meta["captured_at"] = captured_at.strip()
    if anchors:
        meta["anchors"] = merge_anchors_list(None, anchors)
        if not str(meta.get("source_url") or "").strip():
            for row in meta["anchors"]:
                deep = str(row.get("deep_link") or "").strip()
                if deep:
                    meta["source_url"] = deep
                    break
    block = render_snippet_block(
        snippet_id=snippet_id,
        kind=kind,
        text=text,
        image_key=image_key,
        source_url=source_url or str(meta.get("source_url") or ""),
        captured_at=captured_at,
    )
    lines = [f"# {heading}", ""]
    src_line = source_url.strip() or str(meta.get("source_url") or "").strip()
    if src_line:
        lines.extend([f"Source: {src_line}", ""])
    if hint.strip():
        lines.extend([f"Hint: {hint.strip()}", ""])
    lines.append("## Snippets")
    lines.append("")
    lines.append(block.rstrip())
    lines.append("")
    return dump_frontmatter(meta, "\n".join(lines))


def list_collections(cfg: Any, *, store: Any = None) -> list[dict[str, Any]]:
    """Discover existing collection folders under ``capture.enriched_dir``."""
    if store is None:
        store = vault_store_from_config(cfg)
    root = capcfg.enriched_dir(cfg)
    prefix = root.rstrip("/") + "/"
    names: set[str] = set()
    try:
        keys = store.list(root)
    except Exception as exc:
        logger.debug("list collections under %s failed: %s", root, exc)
        keys = []
    for key in keys:
        rel = normalize_vault_key(key)
        if not rel.startswith(prefix):
            continue
        rest = rel.removeprefix(prefix)
        part = rest.split("/", 1)[0].strip()
        if part and part.lower() != "assets":
            names.add(part)
    out = [
        {"id": name, "title": name.replace("-", " ").replace("_", " ").title()}
        for name in sorted(names)
    ]
    return out


def find_inbox_note_for_snippet(
    cfg: Any,
    snippet_id: str,
    *,
    store: Any = None,
) -> str | None:
    """Return an existing inbox path that already holds *snippet_id*, if any."""
    if store is None:
        store = vault_store_from_config(cfg)
    folder = capcfg.inbox_dir(cfg)
    try:
        keys = store.list(folder)
    except Exception:
        return None
    needle = _snippet_marker(snippet_id)
    for key in keys:
        if f"-{snippet_id}.md" in key or key.endswith(f"{snippet_id}.md"):
            try:
                if needle in store.read(key):
                    return key
            except Exception:
                continue
    return None


def save_research_snippets(
    cfg: Any,
    *,
    target: dict[str, Any],
    snippets: list[dict[str, Any]],
    store: Any = None,
) -> dict[str, Any]:
    """Write one inbox atom per snippet. Returns paths and written ids."""
    if store is None:
        store = vault_store_from_config(cfg)
    if not snippets:
        raise CaptureError("snippets must be a non-empty list", status_code=422)

    title = str(target.get("title") or "capture")
    source_url = str(target.get("source_url") or "")
    collection_raw = str(target.get("collection") or "").strip()
    collection = capcfg.slugify(collection_raw) if collection_raw else ""
    hint = str(target.get("hint") or "").strip()

    written: list[str] = []
    skipped: list[str] = []
    paths: list[str] = []
    enrich_paths: list[str] = []
    image_keys: list[str] = []
    written_anchors: list[dict[str, Any]] = []

    for raw in snippets:
        snippet_id = str(raw.get("id") or "").strip()
        if not snippet_id or not _SNIPPET_ID_RE.match(snippet_id):
            raise CaptureError(
                f"invalid snippet id {snippet_id!r}",
                status_code=422,
            )
        skind = str(raw.get("kind") or "text").strip().lower()
        if skind not in SNIPPET_KINDS:
            raise CaptureError(f"unknown snippet kind {skind!r}", status_code=422)

        existing_path = find_inbox_note_for_snippet(cfg, snippet_id, store=store)
        if existing_path:
            skipped.append(snippet_id)
            paths.append(existing_path)
            continue

        key = _research_atom_path(cfg, title=title, snippet_id=snippet_id, store=store)
        if store.exists(key):
            try:
                if note_has_snippet(store.read(key), snippet_id):
                    skipped.append(snippet_id)
                    paths.append(key)
                    continue
            except Exception:
                pass

        image_key = ""
        if skind == "image":
            data = _decode_image(str(raw.get("data_base64") or ""))
            suffix = _image_suffix(str(raw.get("media_type") or ""))
            image_key = f"{assets_dir(cfg)}/{snippet_id}{suffix}"
            store.write_bytes(image_key, data, content_type=_content_type(suffix))
            image_keys.append(image_key)

        anchor = normalize_anchor(raw.get("anchor"), snippet_id=snippet_id)
        atom_anchors = [anchor] if anchor else []
        src = snippet_source_url(raw, source_url)
        body = _render_research_atom(
            title=title,
            snippet_id=snippet_id,
            kind=skind,
            source_url=src,
            collection=collection,
            hint=hint,
            captured_at=str(raw.get("captured_at") or ""),
            image_key=image_key,
            text=str(raw.get("text") or ""),
            anchors=atom_anchors,
        )
        try:
            store.write(key, body)
        except VaultWriteDenied as exc:
            raise CaptureError(str(exc), status_code=403) from exc
        written.append(snippet_id)
        paths.append(key)
        enrich_paths.append(key)
        if atom_anchors:
            written_anchors.extend({**a, "path": key} for a in atom_anchors)

    publish_paths = list(dict.fromkeys([*paths, *image_keys]))
    if publish_paths:
        publish_vault_event(publish_paths, source="capture")

    result: dict[str, Any] = {
        "path": paths[0] if paths else "",
        "paths": paths,
        "enrich_paths": enrich_paths,
        "mode": "research",
        "written": written,
        "skipped": skipped,
        "images": image_keys,
        "enrich": bool(enrich_paths) and capcfg.auto_enrich_enabled(cfg),
    }
    if written_anchors:
        result["anchors"] = written_anchors
    return result


def _replace_captures_section(markdown: str, captures_body: str) -> str:
    body = captures_body.strip()
    if body and not body.endswith("\n"):
        body += "\n"
    match = _CAPTURES_SECTION_RE.search(markdown or "")
    if match:
        start, end = match.span(1)
        return (markdown or "")[:start] + body + (markdown or "")[end:]
    section = f"\n## Captures\n{body or ''}\n"
    notes_idx = (markdown or "").find("## Notes")
    if notes_idx >= 0:
        return (
            (markdown or "")[:notes_idx].rstrip()
            + "\n"
            + section
            + "\n"
            + (markdown or "")[notes_idx:]
        )
    text = markdown or ""
    if text and not text.endswith("\n"):
        text += "\n"
    return text + section


def _entity_link_line(capture_path: str, *, cap_type: str = "", when: str = "") -> str:
    link = normalize_vault_key(capture_path).removesuffix(".md")
    bits = [bit for bit in (cap_type.strip(), when.strip()) if bit]
    suffix = f" — {' · '.join(bits)}" if bits else ""
    return f"- [[{link}]]{suffix}"


def _ensure_entity_note(
    cfg: Any,
    store: Any,
    *,
    collection: str,
    entity: str,
    title: str = "",
    tags: list[str] | None = None,
) -> str:
    coll = capcfg.slugify(collection)
    ent = capcfg.slugify(entity)
    if not coll or not ent:
        raise CaptureError("collection and entity are required to file", status_code=422)
    key = capcfg.entity_path(cfg, collection=coll, entity=ent)
    display = (title or entity or ent).strip() or ent
    try:
        store.read(key)
        return key
    except VaultNotFound:
        pass
    meta = {
        "pawn": "entity",
        "collection": coll,
        "aliases": [],
        "tags": list(tags or []),
        "updated": _now_iso(),
    }
    body = dump_frontmatter(
        meta,
        "\n".join(
            [
                f"# {display}",
                "",
                "## Summary",
                "",
                "",
                "## Captures",
                "",
                "## Notes",
                "",
                "(user-owned)",
                "",
            ]
        ),
    )
    try:
        store.write(key, body)
    except VaultWriteDenied as exc:
        raise CaptureError(str(exc), status_code=403) from exc
    return key


def file_capture(
    cfg: Any,
    *,
    path: str,
    collection: str = "",
    entity: str = "",
    tags: list[str] | None = None,
    ignore: bool = False,
    store: Any = None,
) -> dict[str, Any]:
    """Mark an inbox capture filed or ignored; link entity when filing."""
    if store is None:
        store = vault_store_from_config(cfg)
    key = normalize_vault_key(path)
    if not key:
        raise CaptureError("path is required", status_code=422)
    if not key.lower().endswith(".md"):
        key = f"{key}.md"
    try:
        text = store.read(key)
    except VaultNotFound as exc:
        raise CaptureError(f"note not found: {key}", status_code=404) from exc

    meta, body = parse_frontmatter(text)
    if ignore:
        meta["status"] = "ignored"
        updated = dump_frontmatter(meta, body)
        try:
            store.write(key, updated)
        except VaultWriteDenied as exc:
            raise CaptureError(str(exc), status_code=403) from exc
        publish_vault_event([key], source="capture")
        return {"path": key, "status": "ignored", "entity_path": None}

    coll = (collection or str(meta.get("collection") or "")).strip()
    ent = (entity or str(meta.get("entity") or "")).strip()
    if not coll or not ent:
        raise CaptureError(
            "collection and entity are required to file (set on the note or in the request)",
            status_code=422,
        )
    tag_list = list(tags) if tags is not None else []
    if not tag_list:
        proposed = meta.get("proposed_tags")
        if isinstance(proposed, list):
            tag_list = [str(t).strip() for t in proposed if str(t).strip()]

    entity_key = _ensure_entity_note(
        cfg,
        store,
        collection=coll,
        entity=ent,
        title=ent.replace("-", " ").title(),
        tags=tag_list,
    )
    entity_text = store.read(entity_key)
    link_line = _entity_link_line(
        key,
        cap_type=str(meta.get("type") or ""),
        when=str(meta.get("captured_at") or meta.get("created") or "")[:10],
    )
    section_match = _CAPTURES_SECTION_RE.search(entity_text)
    section_body = section_match.group(1) if section_match else ""
    if normalize_vault_key(key).removesuffix(".md") not in section_body:
        if section_body and not section_body.endswith("\n"):
            section_body += "\n"
        section_body += link_line + "\n"
        entity_text = _replace_captures_section(entity_text, section_body)
        emeta, ebody = parse_frontmatter(entity_text)
        emeta["updated"] = _now_iso()
        emeta["collection"] = capcfg.slugify(coll)
        if tag_list:
            existing_tags = emeta.get("tags") if isinstance(emeta.get("tags"), list) else []
            merged = list(dict.fromkeys([*(str(t) for t in existing_tags), *tag_list]))
            emeta["tags"] = merged
        entity_text = dump_frontmatter(emeta, ebody)
        try:
            store.write(entity_key, entity_text)
        except VaultWriteDenied as exc:
            raise CaptureError(str(exc), status_code=403) from exc

    meta["status"] = "filed"
    meta["collection"] = capcfg.slugify(coll)
    meta["entity"] = capcfg.slugify(ent)
    if tag_list:
        meta["proposed_tags"] = tag_list
    updated = dump_frontmatter(meta, body)
    try:
        store.write(key, updated)
    except VaultWriteDenied as exc:
        raise CaptureError(str(exc), status_code=403) from exc
    publish_vault_event([key, entity_key], source="capture")
    return {"path": key, "status": "filed", "entity_path": entity_key}


def save_snippets(
    cfg: Any,
    *,
    target: dict[str, Any],
    snippets: list[dict[str, Any]],
    store: Any = None,
) -> dict[str, Any]:
    """Create or append ordered snippets. Returns path and written ids."""
    if store is None:
        store = vault_store_from_config(cfg)
    if not snippets:
        raise CaptureError("snippets must be a non-empty list", status_code=422)

    kind = str(target.get("kind") or "new").strip().lower()
    if kind == "research":
        return save_research_snippets(cfg, target=target, snippets=snippets, store=store)
    if kind == "event":
        from pawn_server.core.planner import (  # noqa: PLC0415
            PlannerError,
            upsert_event_from_capture,
        )

        try:
            return upsert_event_from_capture(cfg, target=target, snippets=snippets, store=store)
        except PlannerError as exc:
            raise CaptureError(str(exc), status_code=exc.status_code) from exc

    path = str(target.get("path") or "")
    session_id = str(target.get("session_id") or "")
    title = str(target.get("title") or "")
    source_url = str(target.get("source_url") or "")

    key, mode = _resolve_target(
        cfg,
        store,
        kind=kind,
        path=path,
        session_id=session_id,
        title=title,
        source_url=source_url,
    )

    try:
        existing = store.read(key)
    except VaultNotFound as exc:
        raise CaptureError(f"note not found: {key}", status_code=404) from exc

    written: list[str] = []
    skipped: list[str] = []
    blocks: list[str] = []
    image_keys: list[str] = []
    pending_anchors: list[dict[str, Any]] = []
    register_overlays = kind in OVERLAY_TARGET_KINDS and mode != "annotations"

    for raw in snippets:
        snippet_id = str(raw.get("id") or "").strip()
        if not snippet_id or not _SNIPPET_ID_RE.match(snippet_id):
            raise CaptureError(
                f"invalid snippet id {snippet_id!r}",
                status_code=422,
            )
        skind = str(raw.get("kind") or "text").strip().lower()
        if skind not in SNIPPET_KINDS:
            raise CaptureError(f"unknown snippet kind {skind!r}", status_code=422)

        # Check both full note and annotations body for idempotency
        if note_has_snippet(existing, snippet_id):
            skipped.append(snippet_id)
            continue
        if mode == "annotations" and note_has_snippet(extract_annotations(existing), snippet_id):
            skipped.append(snippet_id)
            continue

        image_key = ""
        if skind == "image":
            data = _decode_image(str(raw.get("data_base64") or ""))
            suffix = _image_suffix(str(raw.get("media_type") or ""))
            image_key = f"{assets_dir(cfg)}/{snippet_id}{suffix}"
            store.write_bytes(image_key, data, content_type=_content_type(suffix))
            image_keys.append(image_key)

        src = snippet_source_url(raw, source_url)
        block = render_snippet_block(
            snippet_id=snippet_id,
            kind=skind,
            text=str(raw.get("text") or ""),
            image_key=image_key,
            source_url=src,
            captured_at=str(raw.get("captured_at") or ""),
        )
        blocks.append(block)
        written.append(snippet_id)
        if register_overlays:
            anchor = normalize_anchor(raw.get("anchor"), snippet_id=snippet_id)
            if anchor:
                pending_anchors.append(anchor)

    updated = existing
    if blocks:
        updated = _append_blocks(existing, blocks, mode=mode)
    if pending_anchors:
        updated = apply_anchors_to_markdown(updated, pending_anchors)
    if blocks or pending_anchors:
        try:
            store.write(key, updated)
        except VaultWriteDenied as exc:
            raise CaptureError(str(exc), status_code=403) from exc
        paths = [key, *image_keys]
        publish_vault_event(paths, source="capture")
    elif image_keys:
        # Should not happen (images only written with blocks), but publish if so
        publish_vault_event(image_keys, source="capture")

    result: dict[str, Any] = {
        "path": key,
        "mode": mode,
        "written": written,
        "skipped": skipped,
        "images": image_keys,
    }
    if pending_anchors:
        result["anchors"] = [{**a, "path": key} for a in pending_anchors]
    return result


def delete_snippet(
    cfg: Any,
    snippet_id: str,
    *,
    path: str,
    store: Any = None,
) -> dict[str, Any]:
    """Strip one snippet block from *path* (and delete its asset if present)."""
    if store is None:
        store = vault_store_from_config(cfg)
    sid = (snippet_id or "").strip()
    if not sid or not _SNIPPET_ID_RE.match(sid):
        raise CaptureError(f"invalid snippet id {sid!r}", status_code=422)
    key = normalize_vault_key(path)
    if not key:
        raise CaptureError("path is required", status_code=422)
    if not key.lower().endswith(".md"):
        key = f"{key}.md"
    try:
        existing = store.read(key)
    except VaultNotFound as exc:
        raise CaptureError(f"note not found: {key}", status_code=404) from exc

    removed_assets: list[str] = []
    for suffix in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
        asset = f"{assets_dir(cfg)}/{sid}{suffix}"
        try:
            store.delete(asset)
            removed_assets.append(asset)
        except Exception:
            continue

    if not note_has_snippet(existing, sid):
        if removed_assets:
            publish_vault_event(removed_assets, source="capture")
        return {
            "path": key,
            "deleted": False,
            "snippet_id": sid,
            "removed_assets": removed_assets,
        }

    # Prefer annotations splice when the marker lives there
    ann = extract_annotations(existing)
    if note_has_snippet(ann, sid):
        updated = replace_annotations(existing, strip_snippet(ann, sid))
    else:
        updated = strip_snippet(existing, sid)

    try:
        store.write(key, updated)
    except VaultWriteDenied as exc:
        raise CaptureError(str(exc), status_code=403) from exc

    publish_vault_event([key, *removed_assets], source="capture")
    return {
        "path": key,
        "deleted": True,
        "snippet_id": sid,
        "removed_assets": removed_assets,
    }
