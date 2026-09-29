"""End-to-end coworker pass for one transcript session or vault note."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from zoneinfo import ZoneInfo

from pawn_agent.core.coworker import db as itemdb
from pawn_agent.core.coworker.extract import extract_items
from pawn_agent.core.coworker.notes import render_item_note, render_today
from pawn_agent.core.coworker.policy import PolicyContext, apply_policy, fingerprint
from pawn_agent.core.coworker.score import score_items
from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.db import create_agent_run, update_agent_run
from pawn_core.goals import Goals, load_goals_text, slugify

logger = logging.getLogger(__name__)


def _model_name(cfg: AgentConfig) -> str:
    return getattr(cfg, "chat_model_id", None) or "coworker"


def _audit(cfg: AgentConfig, command: str, prompt: str, session_id: str) -> str:
    run_id = create_agent_run(
        cfg.db_dsn,
        source="coworker",
        command=command,
        prompt=prompt[:4000],
        session_id=session_id,
        model=_model_name(cfg),
    )
    update_agent_run(cfg.db_dsn, run_id, "running")
    return run_id


def _finish(cfg: AgentConfig, run_id: str, response: str) -> None:
    update_agent_run(cfg.db_dsn, run_id, "completed", response=response[:4000])


def _fail(cfg: AgentConfig, run_id: str, exc: BaseException) -> None:
    update_agent_run(cfg.db_dsn, run_id, "failed", error=str(exc)[:2000])


def _read_goals(cfg: AgentConfig, store: Any) -> Goals:
    try:
        text = store.read(cfg.coworker.goals_path)
    except Exception:
        text = None
    return load_goals_text(text)


def _source_link(cfg: AgentConfig, session_id: str) -> str:
    try:
        from pawn_core.vault_db import get_vault_note  # noqa: PLC0415

        row = get_vault_note(cfg.db_dsn, session_id)
    except Exception:
        return ""
    if row is None or not getattr(row, "key", None):
        return ""
    return f"[[{row.key}]]"


def _local_now(goals: Goals, cfg: AgentConfig) -> datetime:
    name = goals.timezone or cfg.coworker.timezone or "UTC"
    try:
        return datetime.now(ZoneInfo(name))
    except Exception:
        return datetime.now(timezone.utc)


def _start_of_local_day(moment: datetime) -> datetime:
    start = moment.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.astimezone(timezone.utc)


async def process_source(
    cfg: AgentConfig,
    *,
    source_kind: str,
    source_ref: str,
    source_text: str,
    store: Any = None,
    source_link: str = "",
) -> dict[str, Any]:
    """Extract, score, file, and maybe notify for one source."""
    if store is None:
        from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

        store = vault_store_from_config(cfg)
    goals = _read_goals(cfg, store)
    extract_run = _audit(cfg, "extract", source_text[:500], source_ref)
    try:
        extracted = await extract_items(cfg, source_text)
        _finish(cfg, extract_run, f"{len(extracted)} items")
    except Exception as exc:
        _fail(cfg, extract_run, exc)
        raise

    score_run = _audit(cfg, "score", source_ref, source_ref)
    try:
        scored = await score_items(cfg, extracted, goals.active if goals.valid else [])
        _finish(cfg, score_run, f"{len(scored)} scored")
    except Exception as exc:
        _fail(cfg, score_run, exc)
        raise

    now_local = _local_now(goals, cfg)
    triaged = itemdb.triaged_fingerprints(cfg.db_dsn, source_kind, source_ref)
    suppressed = itemdb.suppressed_fingerprints(cfg.db_dsn)
    notified_today = itemdb.count_notified_since(cfg.db_dsn, _start_of_local_day(now_local))
    itemdb.delete_new_for_source(cfg.db_dsn, source_kind, source_ref)

    saved: list[dict[str, Any]] = []
    interrupts = 0
    for item in scored:
        thread = item.get("thread") or ""
        fp = fingerprint(item.get("text") or "", thread)
        if fp in triaged:
            continue
        decision = apply_policy(
            text=item.get("text") or "",
            thread=thread,
            interrupt=bool(item.get("interrupt")) and goals.valid and bool(goals.active),
            fingerprint_value=fp,
            ctx=PolicyContext(
                now=now_local,
                timezone_name=goals.timezone or cfg.coworker.timezone,
                quiet_hours=goals.quiet_hours,
                max_per_day=goals.max_per_day,
                notified_today=notified_today + interrupts,
                ignore=goals.ignore,
                seen_fingerprints=set(),
                suppressed_fingerprints=suppressed,
            ),
        )
        interrupt = decision.interrupt
        if interrupt:
            interrupts += 1
        recurrence = 0
        related: list[str] = []
        try:
            from pawn_agent.core.coworker.link import related_lines  # noqa: PLC0415

            related, recurrence = related_lines(cfg, item.get("text") or "", source_ref)
        except Exception as exc:
            logger.debug("coworker link skipped: %s", exc)
        row = itemdb.insert_item(
            cfg.db_dsn,
            source_kind=source_kind,
            source_ref=source_ref,
            kind=item.get("kind") or "open_question",
            text=item.get("text") or "",
            owner=item.get("owner") or None,
            due=item.get("due") or None,
            quote=item.get("quote") or None,
            thread=thread or None,
            interrupt=interrupt,
            movement=bool(item.get("movement")),
            reason=item.get("reason") or decision.reason,
            fingerprint=fp,
            recurrence=recurrence,
            status="new",
        )
        note_key = f"{cfg.coworker.items_dir.strip('/')}/{row['short_id']}.md"
        note = render_item_note(
            item_id=row["id"],
            short_id=row["short_id"],
            status="new",
            kind=row["kind"],
            text=row["text"],
            thread=thread,
            quote=item.get("quote") or "",
            source_link=source_link,
            reason=row.get("reason") or "",
            interrupt=interrupt,
            related=related,
        )
        try:
            store.write(note_key, note)
            itemdb.update_item(cfg.db_dsn, row["id"], note_key=note_key)
            row["note_key"] = note_key
        except Exception as exc:
            logger.warning("could not write item note %s: %s", note_key, exc)
        if thread:
            try:
                itemdb.upsert_thread(
                    cfg.db_dsn,
                    slug=slugify(thread),
                    name=thread,
                    status="active",
                    movement=bool(item.get("movement")),
                    mention=True,
                )
            except Exception as exc:
                logger.debug("thread upsert skipped: %s", exc)
        if interrupt:
            await _notify_item(cfg, row)
            itemdb.update_item(cfg.db_dsn, row["id"], status="notified")
            row["status"] = "notified"
        saved.append(row)

    try:
        from pawn_core.knowledge_index import index_text  # noqa: PLC0415

        index_text(cfg, source_kind=source_kind, source_ref=source_ref, text=source_text)
    except Exception as exc:
        logger.debug("knowledge index skipped: %s", exc)
    await _maybe_research(cfg, saved, goals, store)
    _rewrite_today(cfg, store)
    return {"source_ref": source_ref, "items": len(saved), "interrupts": interrupts}


async def process_session(
    cfg: AgentConfig,
    session_id: str,
    *,
    store: Any = None,
) -> dict[str, Any]:
    """Run the coworker loop for one diarization session."""
    from pawn_agent.utils.transcript import fetch_transcript  # noqa: PLC0415

    transcript = fetch_transcript(cfg, session_id)
    if transcript.startswith("Error") or transcript.startswith("No transcript"):
        raise ValueError(transcript)
    return await process_source(
        cfg,
        source_kind="session",
        source_ref=session_id,
        source_text=transcript,
        store=store,
        source_link=_source_link(cfg, session_id),
    )


async def process_note(
    cfg: AgentConfig,
    key: str,
    *,
    store: Any = None,
    text: Optional[str] = None,
) -> dict[str, Any]:
    """Run the coworker loop for one vault note."""
    if store is None:
        from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

        store = vault_store_from_config(cfg)
    body = text if text is not None else store.read(key)
    result = await process_source(
        cfg,
        source_kind="note",
        source_ref=key,
        source_text=body,
        store=store,
        source_link=f"[[{key}]]",
    )
    _maybe_idea_companion(cfg, store, key, body)
    return result


def _maybe_idea_companion(cfg: AgentConfig, store: Any, key: str, body: str) -> None:
    lowered = key.lower()
    tagged = "#idea" in body.lower() or "idea" in lowered
    if not tagged and not lowered.startswith("ideas/"):
        return
    from pawn_core.goals import slugify as slug  # noqa: PLC0415

    title = key.rsplit("/", 1)[-1].removesuffix(".md")
    dest = f"{cfg.coworker.ideas_dir.strip('/')}/{slug(title)}.md"
    note = (
        f'---\npawn: idea\nsource: "[[{key}]]"\n---\n'
        f"# {title}\n\n"
        f"Developed from [[{key}]]. Pawn does not edit the original note.\n\n"
        f"## Summary\n\n{body.strip()[:1500]}\n"
    )
    try:
        store.write(dest, note)
    except Exception as exc:
        logger.debug("idea companion skipped: %s", exc)


async def _notify_item(cfg: AgentConfig, item: dict[str, Any]) -> None:
    from pawn_server.core.notify import notify  # noqa: PLC0415

    thread = item.get("thread") or "inbox"
    text = (
        f"Pawn: {item.get('kind')} on {thread}\n"
        f"{item.get('text')}\n"
        f"Reply: file {item['short_id']} | task {item['short_id']} | "
        f"later {item['short_id']} | ignore {item['short_id']}"
    )
    await notify(
        cfg,
        kind="coworker_item",
        text=text,
        link=item.get("note_key") or "",
        item_id=item["id"],
    )


def _rewrite_today(cfg: AgentConfig, store: Any) -> None:
    try:
        rows = itemdb.list_items(cfg.db_dsn, statuses=["new", "notified"], limit=200)
    except Exception as exc:
        logger.warning("today note skipped: %s", exc)
        return
    attention = [row for row in rows if row.get("interrupt")]
    filed = [row for row in rows if not row.get("interrupt")]
    body = render_today({"attention": attention, "filed": filed})
    try:
        store.write(cfg.coworker.today_path, body)
    except Exception as exc:
        logger.warning("could not write today note: %s", exc)


async def _maybe_research(
    cfg: AgentConfig,
    saved: list[dict[str, Any]],
    goals: Goals,
    store: Any,
) -> None:
    """Turn the first interrupt into a research proposal, or run it when allowed."""
    from pawn_agent.core.coworker.autonomy import decide  # noqa: PLC0415

    interrupts = [row for row in saved if row.get("interrupt")]
    if not interrupts:
        return
    decision = decide(cfg, "research", goals=goals)
    if decision.decision == "deny":
        try:
            itemdb.record_decision(
                cfg.db_dsn,
                event_kind="follow_up",
                policy_decision="deny",
                proposed_action="research",
                outcome=decision.reason,
            )
        except Exception:
            logger.debug("follow-up denial not recorded")
        return
    seed = interrupts[0]
    text = f"Look through recent notes and transcripts for more on: {seed.get('text')}"
    fp = fingerprint(text, seed.get("thread") or "")
    row = itemdb.insert_item(
        cfg.db_dsn,
        source_kind="follow_up",
        source_ref=seed["id"],
        kind="proposal",
        text=text,
        thread=seed.get("thread"),
        fingerprint=fp,
        interrupt=decision.decision == "needs_approval",
        reason=decision.reason,
        payload={"prompt": text, "action_kind": "research", "conversation": "coworker:research"},
        status="new",
    )
    if decision.decision == "allow":
        await _run_research(cfg, row, store)
        itemdb.update_item(cfg.db_dsn, row["id"], status="filed")
    else:
        await _notify_item(cfg, row)


async def _run_research(cfg: AgentConfig, row: dict[str, Any], store: Any) -> None:
    from pawn_agent.core.agent_runner import run_agent_turn  # noqa: PLC0415
    from pawn_agent.core.sallm_registry import SallmSessionRegistry  # noqa: PLC0415

    result = await run_agent_turn(
        cfg=cfg,
        registry=SallmSessionRegistry(),
        prompt=row.get("text") or "",
        session_id="coworker:research",
        source="coworker",
        command="research",
        parent_run_id=None,
        depth=1,
        event_id=row["id"],
    )
    thread = row.get("thread") or "Inbox"
    key = f"{cfg.coworker.threads_dir.strip('/')}/{slugify(thread)}.md"
    try:
        existing = ""
        try:
            existing = store.read(key)
        except Exception:
            existing = ""
        from pawn_agent.core.coworker.notes import append_thread_entry  # noqa: PLC0415

        store.write(
            key,
            append_thread_entry(
                existing,
                heading="Research",
                line=result.response[:2000],
                title=thread,
            ),
        )
    except Exception as exc:
        logger.warning("research note skipped: %s", exc)


def rewrite_today(cfg: AgentConfig, store: Any = None) -> None:
    """Public wrapper used by the morning briefing."""
    if store is None:
        from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

        store = vault_store_from_config(cfg)
    _rewrite_today(cfg, store)
