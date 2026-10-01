"""List session screenshots and optionally summarize them (``session_screenshots``)."""

from __future__ import annotations

from pawn_agent.utils.config import AgentConfig
from pawn_core.database import get_engine
from pawn_diarize.core.session_captures import clock_label, load_captures, screenshot_embed


def list_session_screenshots(cfg: AgentConfig, session_id: str) -> str:
    """Print each screenshot's time, output, summary state, and where it lives."""
    shots = [
        cap for cap in load_captures(get_engine(cfg.db_dsn), session_id) if cap.kind == "screenshot"
    ]
    if not shots:
        return f"No screenshots for session {session_id!r}."
    lines: list[str] = []
    for shot in shots:
        summary = (shot.summary or "").strip()
        if not summary:
            state = "not summarized"
        elif summary.lower() == "unchanged":
            state = "unchanged"
        else:
            state = "summarized"
        if shot.vault_key:
            ref = f"![[{screenshot_embed(shot.vault_key, shot.session_id)}]]"
        elif shot.s3_uri:
            ref = shot.s3_uri
        else:
            ref = "not uploaded"
        clock = clock_label(shot.at, shot.at_offset_minutes)
        output = shot.output or "-"
        lines.append(f"{shot.item_id}  {clock}  {output}  {state}  {ref}")
        if state == "summarized":
            lines.append(f"  {summary}")
    return "\n".join(lines)


def summarize_session_screenshots_impl(
    cfg: AgentConfig,
    session_id: str,
    *,
    only_id: str | None = None,
) -> str:
    """Run vision on screenshots that do not yet have a summary, then list them."""
    from pawn_agent.core.screenshot_vision import summarize_session_screenshots

    status = summarize_session_screenshots(
        cfg.db_dsn,
        session_id,
        only_id=only_id,
        s3_cfg=cfg,
        cfg=cfg,
    )
    listing = list_session_screenshots(cfg, session_id)
    return f"{status}\n\n{listing}"
