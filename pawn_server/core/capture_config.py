"""Helpers for ``capture:`` research inbox / enriched paths and model."""

from __future__ import annotations

import re
from typing import Any, Optional

from pawn_core.vault import normalize_vault_key

_SLUG_UNSAFE = re.compile(r"[^\w\-]+", re.UNICODE)
_SLUG_DASH = re.compile(r"-{2,}")


def agent_root(cfg: Any) -> str:
    root = getattr(getattr(cfg, "vault", None), "agent_root", None) or "Pawn"
    return str(root).strip().strip("/") or "Pawn"


def capture_section(cfg: Any) -> Any:
    return getattr(cfg, "capture", None)


def expand_capture_path(cfg: Any, template: str, **extra: str) -> str:
    """Expand ``{agent_root}`` / ``{enriched_dir}`` / named placeholders."""
    root = agent_root(cfg)
    section = capture_section(cfg)
    enriched_tmpl = (
        str(getattr(section, "enriched_dir", "") or "{agent_root}/Research")
        if section is not None
        else "{agent_root}/Research"
    )
    enriched = enriched_tmpl.replace("{agent_root}", root)
    text = (template or "").replace("{agent_root}", root).replace("{enriched_dir}", enriched)
    for key, value in extra.items():
        text = text.replace("{" + key + "}", str(value or "").strip())
    return normalize_vault_key(text)


def inbox_dir(cfg: Any) -> str:
    section = capture_section(cfg)
    tmpl = (
        str(getattr(section, "inbox_dir", "") or "{agent_root}/Captures/Inbox")
        if section is not None
        else "{agent_root}/Captures/Inbox"
    )
    return expand_capture_path(cfg, tmpl)


def enriched_dir(cfg: Any) -> str:
    section = capture_section(cfg)
    tmpl = (
        str(getattr(section, "enriched_dir", "") or "{agent_root}/Research")
        if section is not None
        else "{agent_root}/Research"
    )
    return expand_capture_path(cfg, tmpl)


def entity_path(cfg: Any, *, collection: str, entity: str) -> str:
    section = capture_section(cfg)
    tmpl = (
        str(
            getattr(section, "entity_path_template", "")
            or "{enriched_dir}/{collection}/{entity}.md"
        )
        if section is not None
        else "{enriched_dir}/{collection}/{entity}.md"
    )
    path = expand_capture_path(
        cfg,
        tmpl,
        collection=slugify(collection),
        entity=slugify(entity),
    )
    if not path.lower().endswith(".md"):
        path = f"{path}.md"
    return path


def slugify(value: str, *, fallback: str = "item") -> str:
    text = (value or "").strip().lower().replace(" ", "-")
    text = _SLUG_UNSAFE.sub("-", text)
    text = _SLUG_DASH.sub("-", text).strip("-._")
    return (text[:80].strip("-._") or fallback)[:80]


def resolve_capture_model(cfg: Any) -> Optional[str]:
    """Catalog id for research enrich, or None to keep background default."""
    from pawn_agent.utils.model_catalog import (  # noqa: PLC0415
        catalog_model_or_none,
        default_selection,
        get_background_model,
    )

    section = capture_section(cfg)
    raw = str(getattr(section, "model", "") or "").strip() if section is not None else ""
    chosen = catalog_model_or_none(cfg, raw) if raw else None
    if chosen:
        return chosen
    background = get_background_model(cfg)
    if background:
        return background
    try:
        return default_selection(cfg).catalog_id
    except Exception:
        return None


def auto_enrich_enabled(cfg: Any) -> bool:
    section = capture_section(cfg)
    if section is None:
        return True
    return bool(getattr(section, "auto_enrich", True))


def auto_file_enabled(cfg: Any) -> bool:
    section = capture_section(cfg)
    if section is None:
        return False
    return bool(getattr(section, "auto_file", False))


def capture_instructions(cfg: Any) -> str:
    section = capture_section(cfg)
    if section is None:
        return ""
    return str(getattr(section, "instructions", "") or "").strip()
