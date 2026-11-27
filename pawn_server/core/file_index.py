"""Full-text index of vault files for portal search (``portal_file_index``).

Each visible vault key gets one row with extracted text: Markdown and other
text files as-is (frontmatter stripped), PDF via ``pypdf``, DOCX via
``python-docx``, and images as a one-line caption from the background model
when it is vision-enabled. Rows are keyed by vault key and refreshed when the
S3 etag changes.

Freshness: portal writes index the touched keys right away
(:func:`schedule_index`); everything else (Obsidian Sync, agent writes) is
picked up by :func:`run_index_loop`, which reconciles the whole vault every
``portal.index_interval_seconds``.
"""

from __future__ import annotations

import asyncio
import io
import logging
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional

from sqlalchemy import delete, select

from pawn_core.database import _get_session
from pawn_core.vault import (
    VaultError,
    VaultStat,
    VaultStore,
    is_obsidian_meta,
    normalize_vault_key,
    parse_frontmatter,
)
from pawn_server.core.portal_models import PortalFileIndex, PortalFileTags

logger = logging.getLogger(__name__)

MAX_CHARS = 200_000
_BATCH = 200

TEXT_SUFFIXES = (".md", ".txt", ".json", ".yaml", ".yml", ".csv", ".base", ".canvas", ".log")
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp", ".bmp")
_CAPTION_PROMPT = (
    "Describe this image for a search index in 1-3 plain sentences: what it "
    "shows, any visible title or headings, and any legible text. No preamble."
)

_tasks: set[asyncio.Task] = set()


def _now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def visible_key(key: str) -> bool:
    return bool(key) and not is_obsidian_meta(key) and not key.startswith(".") and "/." not in key


def file_kind(key: str) -> str:
    low = key.lower()
    if low.endswith(".md"):
        return "markdown"
    if low.endswith(TEXT_SUFFIXES):
        return "text"
    if low.endswith(".pdf"):
        return "pdf"
    if low.endswith(".docx"):
        return "docx"
    if low.endswith(IMAGE_SUFFIXES) or low.endswith(".svg"):
        return "image"
    if low.endswith((".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".opus")):
        return "audio"
    return "other"


def frontmatter_tags(meta: dict[str, Any]) -> list[str]:
    """Obsidian ``tags:`` as a list (accepts a list, or a comma/space string)."""
    from pawn_server.core.portal_api import clean_tags  # noqa: PLC0415

    raw = meta.get("tags", meta.get("tag"))
    if raw is None:
        return []
    if isinstance(raw, str):
        items = [p for chunk in raw.split(",") for p in chunk.split()]
    elif isinstance(raw, (list, tuple)):
        items = [str(x) for x in raw if x is not None]
    else:
        items = [str(raw)]
    return clean_tags(items)


def _first_heading(body: str) -> str:
    for line in body.splitlines():
        s = line.strip()
        if s.startswith("#"):
            return s.lstrip("#").strip()[:200]
        if s:
            return ""
    return ""


def _pdf_text(data: bytes) -> str:
    from pypdf import PdfReader  # noqa: PLC0415

    reader = PdfReader(io.BytesIO(data))
    out: list[str] = []
    total = 0
    for page in reader.pages:
        text = page.extract_text() or ""
        out.append(text)
        total += len(text)
        if total > MAX_CHARS:
            break
    return "\n".join(out)


def _docx_text(data: bytes) -> str:
    import docx  # noqa: PLC0415

    document = docx.Document(io.BytesIO(data))
    return "\n".join(p.text for p in document.paragraphs)


def _caption(cfg: Any, key: str, data: bytes) -> str:
    from pawn_server.core.background_llm import describe_image  # noqa: PLC0415

    suffix = Path(key).suffix or ".png"
    handle = tempfile.NamedTemporaryFile(prefix="pawn-index-", suffix=suffix, delete=False)
    try:
        handle.write(data)
        handle.close()
        return describe_image(cfg, Path(handle.name), _CAPTION_PROMPT, session="pawn-file-index")
    finally:
        Path(handle.name).unlink(missing_ok=True)


def extract(key: str, data: bytes, cfg: Any = None, *, captions: bool = True) -> dict[str, Any]:
    """Return ``{kind, title, content, tags, error}`` for one file body."""
    kind = file_kind(key)
    name = key.rsplit("/", 1)[-1]
    out: dict[str, Any] = {"kind": kind, "title": name, "content": "", "tags": [], "error": None}
    try:
        if kind in ("markdown", "text"):
            text = data.decode("utf-8", errors="replace")
            if kind == "markdown":
                meta, body = parse_frontmatter(text)
                out["tags"] = frontmatter_tags(meta)
                title = str(meta.get("title") or "").strip() or _first_heading(body)
                out["title"] = title or name.removesuffix(".md")
                text = body
            out["content"] = text
        elif kind == "pdf":
            out["content"] = _pdf_text(data)
        elif kind == "docx":
            out["content"] = _docx_text(data)
        elif kind == "image" and captions and cfg is not None and not key.lower().endswith(".svg"):
            if bool(getattr(getattr(cfg, "portal", None), "index_images", True)):
                out["content"] = _caption(cfg, key, data)
    except ImportError as exc:
        out["error"] = f"missing extractor: {exc.name or exc}"
    except Exception as exc:  # noqa: BLE001 - one bad file must not stop a pass
        out["error"] = f"{type(exc).__name__}: {exc}"[:500]
    out["content"] = (out["content"] or "").replace("\x00", "")[:MAX_CHARS]
    return out


def _store(cfg: Any) -> VaultStore:
    from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

    return vault_store_from_config(cfg)


def _max_bytes(cfg: Any) -> int:
    return int(getattr(getattr(cfg, "portal", None), "index_max_bytes", 25_000_000))


def index_stats(cfg: Any, store: VaultStore, stats: Iterable[VaultStat]) -> int:
    """(Re)index the given objects; returns how many rows were written."""
    count = 0
    for stat in stats:
        key = normalize_vault_key(stat.key)
        if not visible_key(key):
            continue
        kind = file_kind(key)
        if stat.size > _max_bytes(cfg) or kind in ("audio", "other"):
            result = {"kind": kind, "title": key.rsplit("/", 1)[-1], "content": "", "tags": []}
            result["error"] = "too large" if stat.size > _max_bytes(cfg) else None
        else:
            try:
                data = store.read_bytes(key)
            except VaultError as exc:
                logger.debug("file index: read %s failed: %s", key, exc)
                continue
            result = extract(key, data, cfg)
        modified = stat.last_modified
        if modified is not None and modified.tzinfo is not None:
            modified = modified.astimezone(timezone.utc).replace(tzinfo=None)
        with _get_session(cfg.db_dsn) as db:
            extra = db.get(PortalFileTags, key)
            tags = list(result["tags"])
            for t in (extra.tags if extra else None) or []:
                if t not in tags:
                    tags.append(t)
            row = db.get(PortalFileIndex, key)
            if row is None:
                row = PortalFileIndex(key=key)
                db.add(row)
            row.etag = stat.etag
            row.size = stat.size
            row.modified = modified
            row.kind = result["kind"]
            row.title = result["title"][:500]
            row.content = result["content"]
            row.tags = tags
            row.indexed_at = _now()
            row.error = result.get("error")
        count += 1
    return count


def index_keys(cfg: Any, keys: Iterable[str], store: Optional[VaultStore] = None) -> int:
    """Index specific keys now; keys that no longer exist are dropped."""
    store = store or _store(cfg)
    stats: list[VaultStat] = []
    gone: list[str] = []
    for raw in keys:
        key = normalize_vault_key(raw)
        if not visible_key(key):
            continue
        stat = store.stat(key)
        if stat is None:
            gone.append(key)
        else:
            stats.append(stat)
    if gone:
        with _get_session(cfg.db_dsn) as db:
            db.execute(delete(PortalFileIndex).where(PortalFileIndex.key.in_(gone)))
    return index_stats(cfg, store, stats)


def refresh_tags(cfg: Any, key: str, frontmatter: Optional[list[str]] = None) -> None:
    """Recompute a row's tags after a tag edit (without re-extracting text)."""
    with _get_session(cfg.db_dsn) as db:
        row = db.get(PortalFileIndex, key)
        if row is None:
            return
        extra = db.get(PortalFileTags, key)
        base = frontmatter if frontmatter is not None else []
        tags = list(base)
        for t in (extra.tags if extra else None) or []:
            if t not in tags:
                tags.append(t)
        row.tags = tags


def reconcile(cfg: Any, store: Optional[VaultStore] = None) -> dict[str, int]:
    """Bring the index in line with the vault; returns counts."""
    store = store or _store(cfg)
    stats = [s for s in store.list_meta("") if visible_key(s.key)]
    live = {s.key: s for s in stats}
    with _get_session(cfg.db_dsn) as db:
        known = dict(db.execute(select(PortalFileIndex.key, PortalFileIndex.etag)).all())
        gone = [k for k in known if k not in live]
        for i in range(0, len(gone), _BATCH):
            db.execute(delete(PortalFileIndex).where(PortalFileIndex.key.in_(gone[i : i + _BATCH])))
    changed = [s for k, s in live.items() if known.get(k) != s.etag]
    indexed = index_stats(cfg, store, changed)
    return {"files": len(live), "indexed": indexed, "removed": len(gone)}


def schedule_index(cfg: Any, keys: list[str]) -> None:
    """Index *keys* in a worker thread without blocking the request."""
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    task = loop.create_task(asyncio.to_thread(_safe_index, cfg, list(keys)))
    _tasks.add(task)
    task.add_done_callback(_tasks.discard)


def _safe_index(cfg: Any, keys: list[str]) -> None:
    try:
        index_keys(cfg, keys)
    except Exception as exc:  # noqa: BLE001 - best effort
        logger.warning("file index update failed for %s: %s", keys, exc)


async def run_index_loop(cfg: Any) -> None:
    """Reconcile the vault index forever (``portal.index_interval_seconds``)."""
    interval = int(getattr(getattr(cfg, "portal", None), "index_interval_seconds", 300) or 0)
    if interval <= 0:
        return
    await asyncio.sleep(5)
    while True:
        try:
            counts = await asyncio.to_thread(reconcile, cfg)
            if counts["indexed"] or counts["removed"]:
                logger.info("file index: %s", counts)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - vault/db hiccups retry next pass
            logger.warning("file index reconcile failed: %s", exc)
        await asyncio.sleep(interval)
