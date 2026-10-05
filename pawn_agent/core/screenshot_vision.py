"""Vision deltas for session screenshots.

The automatic pass runs at the end of ``transcribe-diarize`` when
``vault.screenshot_vision`` is on and the background model is flagged
``vision: true``. ``session_screenshots --summarize`` calls the same helper
later, including when the automatic pass was opted out.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Optional

from pawn_agent.core.vision import MAX_IMAGE_BYTES

logger = logging.getLogger(__name__)

_UNCHANGED = "unchanged"
_FIRST = (
    "Describe this screenshot in one or two sentences. "
    "Focus on diagrams, slides, and visible decisions. "
    "If the screen is blank or shows nothing meaningful, reply with exactly: unchanged"
)
_DELTA = (
    "You are looking at two successive screenshots of the same display ({output}). "
    "The first image is earlier; the second is later. "
    "Describe only what changed. If nothing meaningful changed, reply with exactly: unchanged. "
    "One or two sentences. If this is a diagram, name elements that appeared, moved, "
    "or disappeared."
)


def normalize_summary(text: str) -> str:
    """Collapse a no-change reply to ``unchanged``."""
    cleaned = (text or "").strip()
    if not cleaned:
        return _UNCHANGED
    first = cleaned.splitlines()[0].strip().strip("`").rstrip(".").lower()
    if first in {"unchanged", "no change", "no meaningful change", "nothing changed"}:
        return _UNCHANGED
    return cleaned


def maybe_summarize_screenshots(cfg: Any, session_id: str, db_dsn: str) -> None:
    """Best-effort automatic pass. Failures are logged and do not raise."""
    vault = getattr(cfg, "vault", None)
    if vault is not None and not getattr(vault, "screenshot_vision", True):
        logger.info("screenshot vision disabled; skipping session %r", session_id)
        return
    try:
        summarize_session_screenshots(db_dsn, session_id, s3_cfg=cfg)
    except Exception as exc:
        logger.warning(
            "screenshot vision failed for %r (non-fatal): %s",
            session_id,
            exc,
        )


def summarize_session_screenshots(
    db_dsn: str,
    session_id: str,
    *,
    only_id: Optional[str] = None,
    force: bool = False,
    s3_cfg: Any = None,
    cfg: Any = None,
) -> str:
    """Fill missing screenshot summaries. Returns a short status line.

    ``only_id`` limits the model call to one screenshot. The previous image on
    the same output is still sent so the reply describes the change.
    """
    from pawn_agent.utils.config import load_config  # noqa: PLC0415
    from pawn_agent.utils.model_catalog import (  # noqa: PLC0415
        apply_model_selection,
        get_background_model,
        model_is_vision,
    )
    from pawn_core.database import get_engine  # noqa: PLC0415
    from pawn_diarize.core.session_captures import (  # noqa: PLC0415
        load_captures,
        set_summary,
    )

    agent_cfg = cfg if cfg is not None and hasattr(cfg, "agent") else load_config()
    background = get_background_model(agent_cfg)
    if not model_is_vision(agent_cfg, background):
        return "skipped: background model is not vision-enabled"
    if background:
        apply_model_selection(agent_cfg, background)

    source = s3_cfg if s3_cfg is not None else agent_cfg
    engine = get_engine(db_dsn)
    captures = load_captures(engine, session_id)
    shots = [cap for cap in captures if cap.kind == "screenshot"]
    if only_id:
        wanted = only_id.strip()
        if not any(cap.item_id == wanted for cap in shots):
            return f"No screenshot {wanted!r} on session {session_id!r}."
    else:
        wanted = None

    ordered = sorted(shots, key=lambda cap: (cap.at, cap.item_id))
    if wanted:
        target = next(cap for cap in ordered if cap.item_id == wanted)
        group = target.output or ""
        earlier = [
            cap
            for cap in ordered
            if (cap.output or "") == group and (cap.at, cap.item_id) < (target.at, target.item_id)
        ]
        work = ([earlier[-1]] if earlier else []) + [target]
    else:
        work = ordered

    summarized = 0
    skipped = 0
    previous_bytes: Optional[bytes] = None
    current_group: Optional[str] = None
    for shot in work:
        group = shot.output or ""
        if group != current_group:
            previous_bytes = None
            current_group = group
        already = bool((shot.summary or "").strip())
        should = (wanted is None or shot.item_id == wanted) and (force or not already)
        try:
            data = _bytes_for(shot, source)
        except Exception as exc:
            logger.warning(
                "screenshot %s/%s unreadable: %s",
                session_id,
                shot.item_id,
                exc,
            )
            previous_bytes = None
            if should:
                skipped += 1
            continue
        if len(data) > MAX_IMAGE_BYTES:
            logger.warning(
                "screenshot %s/%s is larger than 4MB; leaving it unsummarized",
                session_id,
                shot.item_id,
            )
            previous_bytes = data
            if should:
                skipped += 1
            continue
        if not should:
            previous_bytes = data
            continue
        try:
            summary = _describe(agent_cfg, data, previous_bytes, shot.output)
        except Exception as exc:
            logger.warning(
                "screenshot vision call failed for %s/%s: %s",
                session_id,
                shot.item_id,
                exc,
            )
            previous_bytes = data
            skipped += 1
            continue
        set_summary(engine, session_id, shot.item_id, summary, force=True)
        shot.summary = summary
        previous_bytes = data
        summarized += 1

    if summarized == 0 and skipped == 0:
        return f"No screenshots needed a summary for session {session_id!r}."
    return f"Summarized {summarized} screenshot(s) for session {session_id!r}" + (
        f"; {skipped} skipped." if skipped else "."
    )


def _bytes_for(shot, cfg: Any) -> bytes:
    from pawn_diarize.core.session_captures import read_s3_bytes  # noqa: PLC0415

    if shot.s3_uri:
        return read_s3_bytes(shot.s3_uri, cfg)
    if shot.vault_key:
        from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

        store = vault_store_from_config(cfg)
        return store.read_bytes(shot.vault_key)
    raise RuntimeError("screenshot has no s3_uri or vault_key")


def _describe(cfg: Any, current: bytes, previous: Optional[bytes], output: Optional[str]) -> str:
    from sallm.attachments import image_part  # noqa: PLC0415
    from sallm.llm import complete  # noqa: PLC0415

    from pawn_agent.utils.model_catalog import completion_headers  # noqa: PLC0415

    label = (output or "screen").strip() or "screen"
    prompt = _FIRST if previous is None else _DELTA.format(output=label)
    paths: list[Path] = []
    try:
        if previous is not None:
            paths.append(_write_temp(previous))
        paths.append(_write_temp(current))
        parts: list[dict] = [{"type": "text", "text": prompt}]
        parts.extend(image_part(path) for path in paths)
        selection = cfg.model_selection
        extra: dict = {}
        if selection.api_key:
            extra["api_key"] = selection.api_key
        headers = completion_headers(selection, "pawn-screenshots")
        if headers:
            extra["extra_headers"] = headers
        result = complete(
            model=selection.litellm_model,
            messages=[{"role": "user", "content": parts}],
            api_base=selection.api_base,
            **extra,
        )
    finally:
        for path in paths:
            path.unlink(missing_ok=True)
    return normalize_summary(str(result.get("content") or ""))


def _write_temp(data: bytes) -> Path:
    handle = tempfile.NamedTemporaryFile(prefix="pawn-shot-", suffix=".png", delete=False)
    try:
        handle.write(data)
    finally:
        handle.close()
    return Path(handle.name)
