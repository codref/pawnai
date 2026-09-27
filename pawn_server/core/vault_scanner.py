"""Detect changed vault notes and audio captures for the coworker loop."""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from pawn_agent.core.coworker import db as itemdb
from pawn_agent.core.coworker.capture import capture_audio, is_audio_key
from pawn_core.knowledge_index import content_hash, substantial_change
from pawn_core.vault import is_obsidian_meta, is_under_agent_root, parse_frontmatter

logger = logging.getLogger(__name__)


def _tagged(text: str, tags: list[str]) -> bool:
    meta, body = parse_frontmatter(text or "")
    wanted = {tag.strip().lstrip("#").lower() for tag in tags if tag.strip()}
    found: set[str] = set()
    raw_tags = meta.get("tags")
    if isinstance(raw_tags, str):
        found.add(raw_tags.strip().lstrip("#").lower())
    elif isinstance(raw_tags, list):
        found.update(str(tag).strip().lstrip("#").lower() for tag in raw_tags)
    for tag in wanted:
        if tag in found or f"#{tag}" in (body or "").lower():
            return True
    return False


def _watched(key: str, folders: list[str]) -> bool:
    normalized = key.lstrip("/")
    for folder in folders:
        prefix = folder.strip().strip("/")
        if prefix and (normalized == prefix or normalized.startswith(prefix + "/")):
            return True
    return False


async def run_vault_scanner_tick(cfg: Any, *, store: Any = None) -> dict[str, int]:
    """Process quiet note edits and new audio under the capture folder."""
    if not getattr(cfg.coworker, "enabled", False):
        return {"notes": 0, "audio": 0}
    if store is None:
        from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

        store = vault_store_from_config(cfg)
    notes = await _scan_notes(cfg, store)
    audio = await _scan_audio(cfg, store)
    return {"notes": notes, "audio": audio}


async def _scan_notes(cfg: Any, store: Any) -> int:
    folders = list(cfg.coworker.watch_folders or [])
    keys: list[str] = []
    for folder in folders:
        try:
            keys.extend(store.list(folder))
        except Exception as exc:
            logger.debug("scanner list %s failed: %s", folder, exc)
    processed = 0
    quiet = timedelta(seconds=int(cfg.coworker.note_quiet_seconds or 0))
    now = datetime.now(timezone.utc)
    cap = int(cfg.coworker.max_notes_per_tick or 5)
    for key in keys:
        if processed >= cap:
            break
        if is_obsidian_meta(key) or is_under_agent_root(key, cfg.vault.agent_root):
            continue
        if not await _consider_note(cfg, store, key, now, quiet, require_watch=True):
            continue
        processed += 1
    return processed


async def _consider_note(
    cfg: Any,
    store: Any,
    key: str,
    now: datetime,
    quiet: timedelta,
    *,
    require_watch: bool,
) -> bool:
    try:
        stat = store.stat(key)
        text = store.read(key)
    except Exception as exc:
        logger.debug("scanner skip %s: %s", key, exc)
        return False
    if require_watch and not _watched(key, cfg.coworker.watch_folders):
        if not _tagged(text, list(cfg.coworker.watch_tags or [])):
            return False
    etag = stat.etag if stat else ""
    digest = content_hash(text)
    state = itemdb.get_note_state(cfg.db_dsn, key)
    if state is None or state.etag != etag or state.content_hash != digest:
        itemdb.upsert_note_state(
            cfg.db_dsn,
            key,
            etag=etag,
            content_hash=digest,
            last_seen_at=now,
        )
        return False
    seen = state.last_seen_at
    if seen is not None and seen.tzinfo is None:
        seen = seen.replace(tzinfo=timezone.utc)
    if seen is not None and now - seen < quiet:
        return False
    if state.last_processed_hash == digest:
        return False
    if not substantial_change("", text):
        return False
    from pawn_agent.core.coworker.pipeline import process_note  # noqa: PLC0415

    await process_note(cfg, key, store=store, text=text)
    itemdb.upsert_note_state(
        cfg.db_dsn,
        key,
        last_processed_at=now,
        last_processed_hash=digest,
    )
    try:
        from pawn_core.knowledge_index import index_text  # noqa: PLC0415

        index_text(cfg, source_kind="note", source_ref=key, text=text)
    except Exception as exc:
        logger.debug("index %s skipped: %s", key, exc)
    return True


async def _scan_audio(cfg: Any, store: Any) -> int:
    folder = (cfg.coworker.capture_audio_dir or "").strip()
    if not folder:
        return 0
    try:
        keys = store.list(folder, suffix="")
    except Exception as exc:
        logger.debug("audio list failed: %s", exc)
        return 0
    count = 0
    for key in keys:
        if not is_audio_key(key):
            continue
        state = itemdb.get_note_state(cfg.db_dsn, key)
        if state is not None and state.last_processed_hash == "audio":
            continue
        try:
            await capture_audio(cfg, store, key)
            count += 1
        except Exception as exc:
            logger.warning("capture %s failed: %s", key, exc)
    return count
