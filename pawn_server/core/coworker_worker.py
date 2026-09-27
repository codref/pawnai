"""Morning briefing, weekly review, and vault scan loop."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def _zone(cfg: Any) -> ZoneInfo:
    name = getattr(cfg.coworker, "timezone", None) or "UTC"
    try:
        return ZoneInfo(name)
    except Exception:
        return ZoneInfo("UTC")


def _next_fire(expression: str, after: datetime, zone: ZoneInfo) -> Optional[datetime]:
    try:
        from croniter import croniter  # noqa: PLC0415
    except ImportError:
        logger.warning("croniter is not installed; coworker cron %s disabled", expression)
        return None
    local = after.astimezone(zone)
    nxt = croniter(expression, local).get_next(datetime)
    if nxt.tzinfo is None:
        nxt = nxt.replace(tzinfo=zone)
    return nxt.astimezone(timezone.utc)


async def run_briefing(cfg: Any, *, store: Any = None) -> None:
    """Un-snooze due items, rewrite the daily note, send one Matrix summary."""
    from pawn_agent.core.coworker import db as itemdb  # noqa: PLC0415
    from pawn_agent.core.coworker.loops import (  # noqa: PLC0415
        my_commitments,
        ownerless_decisions,
        recurring_questions,
        stale_threads,
        thread_dict,
    )
    from pawn_agent.core.coworker.pipeline import rewrite_today  # noqa: PLC0415
    from pawn_agent.utils.db import create_agent_run, update_agent_run  # noqa: PLC0415
    from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415
    from pawn_server.core.notify import notify  # noqa: PLC0415

    if store is None:
        store = vault_store_from_config(cfg)
    run_id = create_agent_run(
        cfg.db_dsn,
        source="coworker",
        command="briefing",
        prompt="morning briefing",
        session_id="coworker:daily",
        model=getattr(cfg, "pydantic_model", None) or "coworker",
    )
    update_agent_run(cfg.db_dsn, run_id, "running")
    try:
        woken = itemdb.unsnooze_due(cfg.db_dsn)
        rewrite_today(cfg, store)
        now = datetime.now(timezone.utc)
        items = itemdb.list_open_items(cfg.db_dsn)
        threads = [thread_dict(row) for row in itemdb.list_threads(cfg.db_dsn)]
        loops = []
        loops.extend(
            my_commitments(
                items,
                aliases=list(cfg.coworker.me or []),
                now=now,
                commitment_days=cfg.coworker.commitment_days,
            )
        )
        loops.extend(ownerless_decisions(items))
        loops.extend(recurring_questions(items))
        stale = stale_threads(threads, now=now, stale_days=cfg.coworker.stale_days)
        day = datetime.now(_zone(cfg)).date().isoformat()
        daily_key = f"{cfg.coworker.daily_dir.strip('/')}/{day}.md"
        lines = [f"# {day}", "", f"Unsnoozed: {woken}", "", "## Open loops"]
        if not loops and not stale:
            lines.append("")
            lines.append("_Nothing stalled._")
        for item in loops:
            lines.append(f"- {item.get('kind')}: {item.get('text')}")
        for thread in stale:
            lines.append(f"- stale thread: {thread.get('name')}")
        lines.append("")
        store.write(daily_key, "\n".join(lines))
        await notify(
            cfg,
            kind="coworker_briefing",
            text=f"Pawn morning note is ready ({len(loops)} open loops).",
            link=cfg.coworker.today_path,
        )
        update_agent_run(cfg.db_dsn, run_id, "completed", response=daily_key)
    except Exception as exc:
        update_agent_run(cfg.db_dsn, run_id, "failed", error=str(exc))
        raise


async def run_weekly_review(cfg: Any, *, store: Any = None) -> str:
    """Write ``Pawn/Reviews/{week}.md`` and notify. Never edits Goals.md."""
    from pawn_agent.core.coworker import db as itemdb  # noqa: PLC0415
    from pawn_agent.core.coworker.loops import (  # noqa: PLC0415
        my_commitments,
        ownerless_decisions,
        stale_threads,
        thread_dict,
    )
    from pawn_agent.core.coworker.review import render_review  # noqa: PLC0415
    from pawn_core.goals import load_goals_text  # noqa: PLC0415
    from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415
    from pawn_server.core.notify import notify  # noqa: PLC0415

    if store is None:
        store = vault_store_from_config(cfg)
    try:
        goals_text = store.read(cfg.coworker.goals_path)
    except Exception:
        goals_text = None
    goals = load_goals_text(goals_text)
    now = datetime.now(timezone.utc)
    items = itemdb.list_items(cfg.db_dsn, limit=400)
    threads = [thread_dict(row) for row in itemdb.list_threads(cfg.db_dsn)]
    week = datetime.now(_zone(cfg)).strftime("%G-W%V")
    themes = sorted(
        {
            item.get("text") or ""
            for item in items
            if not item.get("thread") and item.get("kind") != "schedule_proposal"
        }
    )[:12]
    body = render_review(
        week=week,
        goals=goals,
        items=items,
        stale=stale_threads(threads, now=now, stale_days=cfg.coworker.stale_days),
        commitments=my_commitments(
            items,
            aliases=list(cfg.coworker.me or []),
            now=now,
            commitment_days=cfg.coworker.commitment_days,
        ),
        ownerless=ownerless_decisions(items),
        themes=[theme for theme in themes if theme],
    )
    key = f"{cfg.coworker.reviews_dir.strip('/')}/{week}.md"
    store.write(key, body)
    await notify(cfg, kind="coworker_review", text=f"Pawn weekly review {week} is ready.", link=key)
    return key


async def start_coworker(cfg: Any) -> None:
    """Run briefings, the weekly review, and the vault scanner until cancelled."""
    if not cfg.coworker.enabled:
        logger.info("Coworker loop disabled in config")
        return
    zone = _zone(cfg)
    now = datetime.now(timezone.utc)
    next_briefing = _next_fire(cfg.coworker.briefing_cron, now, zone)
    next_weekly = _next_fire(cfg.coworker.weekly_cron, now, zone)
    logger.info(
        "Starting coworker loop | briefing=%s weekly=%s",
        next_briefing,
        next_weekly,
    )
    try:
        while True:
            now = datetime.now(timezone.utc)
            try:
                if next_briefing is not None and now >= next_briefing:
                    await run_briefing(cfg)
                    next_briefing = _next_fire(cfg.coworker.briefing_cron, now, zone)
                if next_weekly is not None and now >= next_weekly:
                    await run_weekly_review(cfg)
                    next_weekly = _next_fire(cfg.coworker.weekly_cron, now, zone)
                from pawn_server.core.vault_scanner import run_vault_scanner_tick  # noqa: PLC0415

                stats = await run_vault_scanner_tick(cfg)
                if any(stats.values()):
                    logger.info("coworker scan %s", stats)
            except Exception as exc:
                logger.error("coworker tick failed: %s", exc, exc_info=True)
            await asyncio.sleep(float(cfg.vault_watcher.poll_interval_seconds))
    except asyncio.CancelledError:
        logger.info("Coworker loop cancelled")
        raise
