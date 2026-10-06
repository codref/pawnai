"""Triage actions shared by the vault watcher, Matrix, and the HTTP API."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from pawn_agent.core.coworker import db as itemdb
from pawn_agent.core.coworker.notes import append_thread_entry, render_item_note, thread_note_key
from pawn_agent.utils.config import AgentConfig
from pawn_core.vault_config import vault_store_from_config

logger = logging.getLogger(__name__)

_COMMAND_RE = re.compile(
    r"^(file|task|later|ignore|approve|reject)\s+(\S+)(?:\s+(.+))?$",
    re.IGNORECASE,
)

ACTIONS = frozenset({"file", "task", "later", "ignore", "approve", "reject"})


def parse_coworker_command(text: str) -> Optional[tuple[str, str, Optional[str]]]:
    """Parse ``file|task|later|ignore|approve|reject <id> [arg]``."""
    match = _COMMAND_RE.match((text or "").strip())
    if not match:
        return None
    action = match.group(1).lower()
    item_id = match.group(2).strip()
    arg = (match.group(3) or "").strip() or None
    return action, item_id, arg


def _store(cfg: AgentConfig, store: Any) -> Any:
    if store is not None:
        return store
    return vault_store_from_config(cfg)


def _rewrite_note(store: Any, item: dict[str, Any], *, status: str, action: str = "") -> None:
    key = item.get("note_key")
    if not key or store is None:
        return
    body = render_item_note(
        item_id=item["id"],
        short_id=item["short_id"],
        status=status,
        kind=item.get("kind") or "",
        text=item.get("text") or "",
        thread=item.get("thread") or "",
        quote=item.get("quote") or "",
        reason=item.get("reason") or "",
        interrupt=bool(item.get("interrupt")),
        action=action,
    )
    store.write(key, body)


def _append(store: Any, cfg: AgentConfig, thread: str, heading: str, line: str) -> None:
    if not thread or store is None:
        return
    key = thread_note_key(cfg.coworker.threads_dir, thread)
    try:
        existing = store.read(key)
    except Exception:
        existing = ""
    store.write(key, append_thread_entry(existing, heading=heading, line=line, title=thread))


async def apply_action(
    cfg: AgentConfig,
    item_id: str,
    action: str,
    arg: Optional[str] = None,
    *,
    registry: Any = None,
    store: Any = None,
) -> str:
    """Apply a triage action. Returns a one-line receipt."""
    action = (action or "").strip().lower()
    if action not in ACTIONS:
        return f"Unknown action {action!r}."
    item = itemdb.get_item(cfg.db_dsn, item_id)
    if item is None:
        return f"No item {item_id}."
    vault = None
    try:
        vault = _store(cfg, store)
    except Exception as exc:
        logger.warning("coworker action has no vault: %s", exc)

    if action == "file":
        _append(vault, cfg, item.get("thread") or "Inbox", "Filed", item.get("text") or "")
        updated = itemdb.update_item(cfg.db_dsn, item["id"], status="filed")
        _rewrite_note(vault, updated or item, status="filed")
        return f"Filed {item['short_id']}."

    if action == "task":
        _append(vault, cfg, item.get("thread") or "Inbox", "Open loops", item.get("text") or "")
        if registry is not None:
            await _spawn_task(cfg, item, registry)
        updated = itemdb.update_item(cfg.db_dsn, item["id"], status="task")
        _rewrite_note(vault, updated or item, status="task")
        return f"Task captured for {item['short_id']}."

    if action == "later":
        until = _parse_until(arg)
        updated = itemdb.update_item(cfg.db_dsn, item["id"], status="snoozed", snooze_until=until)
        _rewrite_note(vault, updated or item, status="snoozed")
        return f"Snoozed {item['short_id']} until {until.date().isoformat()}."

    if action == "ignore":
        itemdb.add_suppression(cfg.db_dsn, item["fingerprint"])
        updated = itemdb.update_item(cfg.db_dsn, item["id"], status="dismissed")
        _rewrite_note(vault, updated or item, status="dismissed")
        return f"Ignored {item['short_id']}."

    if action == "approve" and item.get("kind") == "proposal":
        from pawn_agent.core.coworker.pipeline import _run_research  # noqa: PLC0415

        await _run_research(cfg, item, vault)
        updated = itemdb.update_item(cfg.db_dsn, item["id"], status="filed")
        _rewrite_note(vault, updated or item, status="filed")
        return f"Ran research for {item['short_id']}."

    if action == "approve" and item.get("kind") == "people_update":
        return await _approve_people_update(cfg, item, vault)

    if action == "reject" and item.get("kind") == "people_update":
        updated = itemdb.update_item(cfg.db_dsn, item["id"], status="dismissed")
        _rewrite_note(vault, updated or item, status="dismissed")
        return f"Rejected people update {item['short_id']}."

    if action in {"approve", "reject"}:
        return _resolve_proposal(cfg, item, action, vault)

    return f"Unknown action {action!r}."


def _parse_until(arg: Optional[str]) -> datetime:
    now = datetime.now(timezone.utc)
    if arg:
        text = arg.strip()
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M:%S"):
            try:
                parsed = datetime.strptime(text, fmt)
                if parsed.tzinfo is None:
                    parsed = parsed.replace(tzinfo=timezone.utc)
                return parsed.astimezone(timezone.utc)
            except ValueError:
                continue
    return now + timedelta(days=1)


async def _approve_people_update(cfg: AgentConfig, item: dict[str, Any], vault: Any) -> str:
    """Apply a proposed People/ note update payload."""
    from pawn_agent.core.people.refresh import apply_people_updates  # noqa: PLC0415

    payload = item.get("payload") or {}
    updates = payload.get("updates") or []
    if not isinstance(updates, list) or not updates:
        return f"People update {item['short_id']} has no payload."
    keys = apply_people_updates(cfg, updates, store=vault)
    updated = itemdb.update_item(cfg.db_dsn, item["id"], status="filed")
    _rewrite_note(vault, updated or item, status="filed")
    return f"Applied people update {item['short_id']} ({len(keys)} note(s))."


async def _spawn_task(cfg: AgentConfig, item: dict[str, Any], registry: Any) -> None:
    from pawn_server.core.jobs import create_ask_job  # noqa: PLC0415

    instruction = item.get("text") or "Follow up"
    await create_ask_job(
        cfg,
        registry=registry,
        job_id=item["id"],
        instruction=instruction,
        note_path=item.get("note_key"),
        conversation=f"coworker:{item['short_id']}",
    )


def _resolve_proposal(cfg: AgentConfig, item: dict[str, Any], action: str, vault: Any) -> str:
    payload = item.get("payload") or {}
    proposal_id = str(payload.get("proposal_id") or "")
    if item.get("kind") != "schedule_proposal" or not proposal_id:
        return f"{item['short_id']} is not a schedule proposal."
    from pawn_agent.core.scheduler import AgentSchedulerService  # noqa: PLC0415

    service = AgentSchedulerService(
        cfg.db_dsn, default_timezone=cfg.agent_scheduler.default_timezone
    )
    if action == "approve":
        service.approve_proposal(proposal_id, reviewed_by="coworker")
        status = "filed"
        receipt = f"Approved schedule {item['short_id']}."
    else:
        service.reject_proposal(proposal_id, reviewed_by="coworker")
        status = "dismissed"
        receipt = f"Rejected schedule {item['short_id']}."
    updated = itemdb.update_item(cfg.db_dsn, item["id"], status=status)
    _rewrite_note(vault, updated or item, status=status)
    return receipt
