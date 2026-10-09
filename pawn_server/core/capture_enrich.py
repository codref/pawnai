"""Research capture enrich — sallm ReAct turn over one inbox atom."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional
from uuid import uuid4

from pawn_agent.core.agent_runner import run_agent_turn
from pawn_agent.core.sallm_registry import SallmSessionRegistry
from pawn_agent.core.vision import TurnImage, mime_for
from pawn_core.vault import VaultNotFound, normalize_vault_key, parse_frontmatter
from pawn_core.vault_config import vault_store_from_config
from pawn_server.core import capture_config as capcfg

logger = logging.getLogger(__name__)


def _image_asset_from_note(text: str) -> Optional[str]:
    meta, body = parse_frontmatter(text)
    if str(meta.get("kind") or "").strip().lower() != "image":
        return None
    for line in body.splitlines():
        line = line.strip()
        if line.startswith("![[") and line.endswith("]]"):
            return line[3:-2].strip()
    return None


def build_enrich_prompt(cfg: Any, path: str, note_text: str) -> str:
    """Fixed brief for the ``research_capture`` skill."""
    inbox = capcfg.inbox_dir(cfg)
    enriched = capcfg.enriched_dir(cfg)
    section = capcfg.capture_section(cfg)
    entity_tmpl = (
        str(getattr(section, "entity_path_template", "") or "") if section is not None else ""
    ) or "{enriched_dir}/{collection}/{entity}.md"
    auto_file = capcfg.auto_file_enabled(cfg)
    extra = capcfg.capture_instructions(cfg)
    meta, _ = parse_frontmatter(note_text)
    collection = str(meta.get("collection") or "").strip()
    hint = str(meta.get("hint") or "").strip()
    source_url = str(meta.get("source_url") or "").strip()
    lines = [
        "Active skill: research_capture.",
        "Enrich this research capture inbox note and file it in the right place.",
        "",
        f"Capture note path: {path}",
        f"Inbox dir: {inbox}",
        f"Enriched dir: {enriched}",
        f"Entity path template: {entity_tmpl}",
        f"auto_file: {'true' if auto_file else 'false'}",
    ]
    if collection:
        lines.append(f"User collection hint: {collection}")
    else:
        lines.append("User collection hint: (none — propose a short collection slug)")
    if hint:
        lines.append(f"User hint: {hint}")
    if source_url:
        lines.append(f"Source URL: {source_url}")
    if extra:
        lines.extend(["", "Extra instructions:", extra])
    lines.extend(
        [
            "",
            "Steps:",
            f"1) note_read --path {path}",
            f"2) note_search --folder {enriched} to see existing collections/entities.",
            "3) If collection is empty, propose a short slug (e.g. movies, recipes); "
            "prefer an existing folder under the enriched dir when it fits.",
            "4) Find or create the entity note under the enriched dir "
            "(pawn: entity, ## Captures, ## Notes left for the user).",
            "5) Update the capture note frontmatter: collection, entity, type, "
            "caption, proposed_tags, enriched_at (ISO UTC). Keep the snippet block intact.",
            "6) If auto_file is true and the match is clear: append a wikilink under "
            "the entity ## Captures section and set capture status: filed. "
            "Otherwise set status: proposed and stop — do not invent confident filing.",
            "7) Only write under the inbox dir and enriched dir. Never dump tool errors "
            "into notes. Answer with a one-line summary of collection/entity/status.",
        ]
    )
    return "\n".join(lines)


async def run_capture_enrich(
    cfg: Any,
    *,
    path: str,
    registry: SallmSessionRegistry,
    store: Any = None,
) -> str:
    """Run one enrich turn. Returns the agent reply text."""
    if store is None:
        store = vault_store_from_config(cfg)
    key = normalize_vault_key(path)
    if not key.lower().endswith(".md"):
        key = f"{key}.md"
    try:
        note_text = await asyncio.to_thread(store.read, key)
    except VaultNotFound as exc:
        raise FileNotFoundError(f"capture note not found: {key}") from exc

    meta, _ = parse_frontmatter(note_text)
    status = str(meta.get("status") or "").strip().lower()
    if status in {"filed", "ignored"}:
        return f"skipped: status is {status}"

    prompt = build_enrich_prompt(cfg, key, note_text)
    model = capcfg.resolve_capture_model(cfg)
    images: list[TurnImage] = []
    asset = _image_asset_from_note(note_text)
    if asset:
        try:
            data = await asyncio.to_thread(store.read_bytes, asset)
            images.append(
                TurnImage(
                    filename=asset.rsplit("/", 1)[-1],
                    media_type=mime_for(asset),
                    data=bytes(data),
                    role="question",
                )
            )
        except Exception as exc:
            logger.info("capture enrich: could not load image %s: %s", asset, exc)

    conv = f"note:{key}"
    result = await run_agent_turn(
        cfg=cfg,
        registry=registry,
        prompt=prompt,
        session_id=conv,
        model=model,
        source="capture",
        command="capture_enrich",
        images=images or None,
    )
    return result.response or ""


def new_enrich_job_id() -> str:
    return str(uuid4())
