"""Update research-capture inbox note frontmatter safely (parse + dump)."""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any, Optional, Sequence

from pawn_agent.utils.config import AgentConfig
from pawn_core.vault import (
    VaultNotFound,
    VaultWriteDenied,
    dump_frontmatter,
    normalize_vault_key,
    parse_frontmatter,
)
from pawn_core.vault_config import vault_store_from_config

_SLUG_UNSAFE = re.compile(r"[^\w\-]+", re.UNICODE)
_SLUG_DASH = re.compile(r"-{2,}")

_UPDATABLE = frozenset(
    {
        "collection",
        "entity",
        "type",
        "caption",
        "proposed_tags",
        "status",
        "enriched_at",
        "hint",
    }
)
_STATUSES = frozenset({"inbox", "proposed", "filed", "ignored"})


def slugify(value: str, *, fallback: str = "item") -> str:
    text = (value or "").strip().lower().replace(" ", "-")
    text = _SLUG_UNSAFE.sub("-", text)
    text = _SLUG_DASH.sub("-", text).strip("-._")
    return (text[:80].strip("-._") or fallback)[:80]


def _now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _parse_tags(raw: Optional[str | Sequence[str]]) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, (list, tuple)):
        return [str(t).strip().lstrip("#") for t in raw if str(t).strip()]
    text = str(raw).strip()
    if not text:
        return []
    if text.startswith("["):
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return [str(t).strip().lstrip("#") for t in data if str(t).strip()]
        except json.JSONDecodeError:
            pass
    return [part.strip().lstrip("#") for part in text.split(",") if part.strip()]


def capture_update_impl(
    cfg: AgentConfig,
    path: str,
    *,
    collection: Optional[str] = None,
    entity: Optional[str] = None,
    type: Optional[str] = None,  # noqa: A002 — CLI field name
    caption: Optional[str] = None,
    proposed_tags: Optional[str | Sequence[str]] = None,
    status: Optional[str] = None,
    enriched_at: Optional[str] = None,
    hint: Optional[str] = None,
    set_enriched_now: bool = False,
) -> str:
    """Merge frontmatter fields on a capture note; keep body intact.

    Uses :func:`dump_frontmatter` so values with ``:`` stay valid YAML for Obsidian.
    """
    store = vault_store_from_config(cfg)
    key = normalize_vault_key(path)
    if not key:
        return "Error: empty vault path"
    if not key.lower().endswith(".md"):
        key = f"{key}.md"
    try:
        text = store.read(key)
    except VaultNotFound:
        return f"Error: note not found: {key!r}"
    except Exception as exc:
        return f"Error reading {key!r}: {exc}"

    meta, body = parse_frontmatter(text)
    if not meta and body.lstrip().startswith("---"):
        # Opportunistic: leading junk before a fence — try strip to first ---
        idx = body.find("---")
        if idx > 0:
            meta, body = parse_frontmatter(body[idx:])

    pawn_val = str(meta.get("pawn") or "").strip()
    if pawn_val and pawn_val != "capture":
        return f"Error: {key!r} is not a capture note (pawn={pawn_val!r})"

    meta["pawn"] = "capture"
    if "tags" not in meta:
        meta["tags"] = ["pawn/capture"]

    updates: dict[str, Any] = {}
    if collection is not None and str(collection).strip():
        updates["collection"] = slugify(str(collection))
    if entity is not None and str(entity).strip():
        updates["entity"] = slugify(str(entity))
    if type is not None:
        updates["type"] = str(type).strip()
    if caption is not None:
        updates["caption"] = str(caption)
    if proposed_tags is not None:
        updates["proposed_tags"] = _parse_tags(proposed_tags)
    if status is not None and str(status).strip():
        st = str(status).strip().lower()
        if st not in _STATUSES:
            return f"Error: status must be one of {sorted(_STATUSES)}"
        updates["status"] = st
    if enriched_at is not None and str(enriched_at).strip():
        updates["enriched_at"] = str(enriched_at).strip()
    elif set_enriched_now or any(
        k in updates for k in ("collection", "entity", "type", "caption", "proposed_tags")
    ):
        updates["enriched_at"] = _now_iso()
    if hint is not None:
        updates["hint"] = str(hint)

    if not updates:
        return f"No changes for {key}"

    for key_name, value in updates.items():
        if key_name in _UPDATABLE:
            meta[key_name] = value

    updated = dump_frontmatter(meta, body)
    try:
        store.write(key, updated)
    except VaultWriteDenied as exc:
        return f"Error: write denied: {exc}"
    except Exception as exc:
        return f"Error writing {key!r}: {exc}"

    bits = [f"{k}={v!r}" for k, v in updates.items()]
    return f"Updated {key}: " + ", ".join(bits) + "\n"
