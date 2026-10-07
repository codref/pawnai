"""Triage actions shared by the vault watcher, Matrix, and the HTTP API."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from pawn_agent.core.coworker import db as itemdb
from pawn_agent.core.coworker.notes import append_thread_entry, render_item_note, thread_note_key
from pawn_agent.utils.config import AgentConfig
from pawn_core.goals import slugify
from pawn_core.vault import dump_frontmatter
from pawn_core.vault_config import vault_store_from_config

logger = logging.getLogger(__name__)

_COMMAND_RE = re.compile(
    r"^(file|task|later|ignore|approve|reject|todo|delete)\s+(\S+)(?:\s+(.+))?$",
    re.IGNORECASE,
)

ACTIONS = frozenset({"file", "task", "later", "ignore", "approve", "reject", "todo", "delete"})

_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


def parse_coworker_command(text: str) -> Optional[tuple[str, str, Optional[str]]]:
    """Parse ``file|task|later|ignore|approve|reject|todo|delete <id> [arg]``."""
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


def _delete_note(store: Any, item: dict[str, Any]) -> None:
    key = item.get("note_key")
    if not key or store is None:
        return
    try:
        store.delete(key)
    except Exception as exc:
        logger.warning("could not delete item note %s: %s", key, exc)


def _append(store: Any, cfg: AgentConfig, thread: str, heading: str, line: str) -> None:
    if not thread or store is None:
        return
    key = thread_note_key(cfg.coworker.threads_dir, thread)
    try:
        existing = store.read(key)
    except Exception:
        existing = ""
    store.write(key, append_thread_entry(existing, heading=heading, line=line, title=thread))


def _task_stem(title: str) -> str:
    cleaned = _UNSAFE.sub(" ", title or "").replace("\n", " ")
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip(".")
    if len(cleaned) > 80:
        cleaned = cleaned[:80].strip()
    return cleaned or "task"


def _todo_folder(cfg: AgentConfig) -> str:
    root = (cfg.vault.agent_root or "Pawn").strip().strip("/") or "Pawn"
    external = (cfg.tasknotes.external_tasks_dir or "TaskNotes/Tasks").strip().strip("/")
    if external.startswith(root + "/"):
        return external
    return f"{root}/{external}"


def _write_todo_note(store: Any, cfg: AgentConfig, item: dict[str, Any]) -> Optional[str]:
    """Create a TaskNotes note for *item*. Returns the vault key."""
    if store is None:
        return None
    title = " ".join((item.get("text") or "Follow up").split())
    folder = _todo_folder(cfg)
    stem = _task_stem(title)
    path = f"{folder}/{stem}.md"
    n = 2
    while True:
        try:
            store.read(path)
        except Exception:
            break
        path = f"{folder}/{stem} {n}.md"
        n += 1
    source = item.get("note_key") or ""
    link = f"[[{source[:-3]}]]" if source.endswith(".md") else (f"[[{source}]]" if source else "")
    body = (item.get("text") or "").strip()
    meta: dict[str, Any] = {
        "tags": ["task"],
        "title": title,
        "status": "open",
        "priority": "normal",
    }
    parts = []
    if link:
        parts.append(f"Source: {link}")
        parts.append("")
    parts.append(body)
    store.write(path, dump_frontmatter(meta, "\n".join(parts) + "\n"))
    return path


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

    if action == "todo":
        path = _write_todo_note(vault, cfg, item)
        payload = dict(item.get("payload") or {})
        if path:
            payload["todo_path"] = path
        updated = itemdb.update_item(cfg.db_dsn, item["id"], status="task", payload=payload or None)
        _rewrite_note(vault, updated or item, status="task")
        if path:
            return f"TODO {path} for {item['short_id']}."
        return f"TODO recorded for {item['short_id']} (no vault)."

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

    if action == "delete":
        itemdb.add_suppression(cfg.db_dsn, item["fingerprint"])
        itemdb.update_item(cfg.db_dsn, item["id"], status="dismissed")
        _delete_note(vault, item)
        return f"Deleted {item['short_id']}."

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


async def delete_items(
    cfg: AgentConfig,
    *,
    ids: Optional[list[str]] = None,
    all_open: bool = False,
    kind: Optional[str] = None,
    q: Optional[str] = None,
    registry: Any = None,
    store: Any = None,
) -> dict[str, Any]:
    """Delete many items. Returns ``{deleted, receipts}``."""
    del registry  # unused; kept for call-site symmetry with apply_action
    target_ids: list[str] = []
    if all_open:
        target_ids = itemdb.list_item_ids(
            cfg.db_dsn,
            statuses=list(itemdb.OPEN_STATUSES),
            kind=kind,
            q=q,
        )
    elif ids:
        # Resolve short_ids to canonical ids via get_item.
        seen: set[str] = set()
        for raw in ids:
            item = itemdb.get_item(cfg.db_dsn, raw)
            if item and item["id"] not in seen:
                seen.add(item["id"])
                target_ids.append(item["id"])
    receipts: list[str] = []
    for item_id in target_ids:
        receipts.append(await apply_action(cfg, item_id, "delete", store=store))
    return {"deleted": len(target_ids), "receipts": receipts}


def item_note_key(
    cfg: AgentConfig, *, short_id: str, text: str, created_at: Optional[datetime] = None
) -> str:
    """Readable vault key for a new item note."""
    day = (created_at or datetime.now(timezone.utc)).date().isoformat()
    slug = slugify(" ".join((text or "").split())[:40])
    if slug == "thread":
        slug = "item"
    folder = cfg.coworker.items_dir.strip("/")
    return f"{folder}/{day}-{slug}-{short_id}.md"


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
