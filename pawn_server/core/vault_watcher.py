"""Poll vault task notes and run agent lifecycle (slow path)."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Optional

from pawn_agent.core.sallm_registry import SallmSessionRegistry
from pawn_agent.utils.db import (
    claim_vault_task,
    get_vault_task,
    get_vault_task_by_key,
    update_vault_task,
    upsert_vault_task,
)
from pawn_core.vault import VaultNotFound, normalize_vault_key
from pawn_core.vault_config import vault_store_from_config
from pawn_server.core.vault_protocol import (
    conversation_id_for_note,
    instruction_hash,
    parse_task_note,
)
from pawn_server.core.vault_tasks import approve_vault_task, execute_vault_task

logger = logging.getLogger(__name__)

_STATUSES_SKIP_ETAG = frozenset({"running", "review", "done", "blocked", "claimed"})


def _needs_action(
    note_status: str,
    approved: bool,
    *,
    db_status: str | None = None,
    indexed: bool = False,
) -> bool:
    if note_status == "todo":
        return True
    if note_status == "review" and approved:
        return True
    if note_status == "done" and db_status == "review" and not indexed:
        return True
    return False


async def run_vault_watcher_tick(
    cfg: Any,
    *,
    registry: Optional[SallmSessionRegistry] = None,
) -> dict[str, int]:
    """One watcher tick: scan task notes, execute todo, approve review, dismiss done."""
    active = registry or SallmSessionRegistry()
    store = vault_store_from_config(cfg)
    agent_root = normalize_vault_key(cfg.vault.agent_root).rstrip("/") or "Pawn"
    tasks_prefix = f"{agent_root}/Tasks"

    try:
        keys = await asyncio.to_thread(store.list, tasks_prefix)
    except Exception as exc:
        logger.error("vault_watcher list failed: %s", exc, exc_info=True)
        return {"executed": 0, "approved": 0, "dismissed": 0}

    executed = 0
    approved = 0
    dismissed = 0
    max_claims = max(1, int(cfg.vault_watcher.max_claims_per_tick))

    for task_key in keys:
        if executed + approved + dismissed >= max_claims:
            break
        try:
            stat = await asyncio.to_thread(store.stat, task_key)
            body = await asyncio.to_thread(store.read, task_key)
            parsed = parse_task_note(body)
        except VaultNotFound:
            continue
        except ValueError:
            continue
        except Exception as exc:
            logger.debug("vault_watcher skip %s: %s", task_key, exc)
            continue

        task_id = (parsed.get("id") or "").strip()
        if not task_id:
            continue

        note_status = parsed.get("status") or "todo"
        etag = stat.etag if stat else None
        existing = get_vault_task(cfg.db_dsn, task_id) or get_vault_task_by_key(
            cfg.db_dsn, task_key
        )
        if (
            existing
            and etag
            and existing.etag == etag
            and not _needs_action(
                note_status,
                bool(parsed.get("approved")),
                db_status=existing.status,
                indexed=existing.indexed_at is not None,
            )
            and existing.status in _STATUSES_SKIP_ETAG
        ):
            continue

        if note_status == "review" and parsed.get("approved"):
            eff_id = existing.id if existing else task_id
            if existing is None:
                eff_id = upsert_vault_task(
                    cfg.db_dsn,
                    task_id=task_id,
                    key=task_key,
                    instruction_hash=instruction_hash(parsed.get("instruction") or ""),
                    conversation_id=parsed.get("conversation")
                    or conversation_id_for_note(task_key),
                    instruction_text=parsed.get("instruction"),
                    note_path=parsed.get("note_path"),
                    etag=etag,
                    status="review",
                )
            await approve_vault_task(
                cfg,
                eff_id,
                registry=active,
                store=store,
                result_text=parsed.get("result"),
            )
            approved += 1
            continue

        if note_status == "done":
            if (
                existing
                and existing.status == "review"
                and existing.indexed_at is None
            ):
                from pawn_server.core.jobs import dismiss_job  # noqa: PLC0415

                try:
                    await dismiss_job(cfg, existing.id)
                    dismissed += 1
                except Exception as exc:
                    logger.debug("vault_watcher dismiss %s: %s", existing.id, exc)
            continue

        if note_status != "todo":
            continue

        instruction = parsed.get("instruction") or ""
        ih = instruction_hash(instruction)
        conv = parsed.get("conversation") or conversation_id_for_note(task_key)
        effective_id = upsert_vault_task(
            cfg.db_dsn,
            task_id=task_id,
            key=task_key,
            instruction_hash=ih,
            conversation_id=conv,
            instruction_text=instruction,
            note_path=parsed.get("note_path"),
            etag=etag,
            via="vault",
            status="queued",
        )
        if not claim_vault_task(cfg.db_dsn, effective_id):
            continue
        if etag:
            update_vault_task(cfg.db_dsn, effective_id, etag=etag)
        await execute_vault_task(
            cfg,
            effective_id,
            registry=active,
            store=store,
            write_result_to_vault=True,
        )
        executed += 1

    acted = await _apply_item_actions(cfg, store, registry=active)
    return {
        "executed": executed,
        "approved": approved,
        "dismissed": dismissed,
        "items": acted,
    }


async def _apply_item_actions(cfg: Any, store: Any, *, registry: Any) -> int:
    """Apply ``action:`` frontmatter on coworker item notes."""
    coworker = getattr(cfg, "coworker", None)
    if coworker is None or not getattr(coworker, "enabled", False):
        return 0
    items_dir = getattr(coworker, "items_dir", "") or ""
    if not items_dir:
        return 0
    from pawn_agent.core.coworker.actions import apply_action  # noqa: PLC0415
    from pawn_agent.core.coworker.notes import TERMINAL_STATUSES, parse_item_note  # noqa: PLC0415

    try:
        keys = await asyncio.to_thread(store.list, items_dir)
    except Exception as exc:
        logger.debug("item scan skipped: %s", exc)
        return 0
    acted = 0
    for key in keys:
        try:
            body = await asyncio.to_thread(store.read, key)
            parsed = parse_item_note(body)
        except Exception:
            continue
        action = parsed.get("action") or ""
        if not action or parsed.get("status") in TERMINAL_STATUSES:
            continue
        item_id = parsed.get("id") or parsed.get("short_id")
        if not item_id:
            continue
        await apply_action(cfg, item_id, action, registry=registry, store=store)
        acted += 1
    return acted


async def start_vault_watcher(
    cfg: Any,
    *,
    registry: Optional[SallmSessionRegistry] = None,
) -> None:
    """Run the vault task watcher loop until cancelled."""
    if not cfg.vault_watcher.enabled:
        logger.info("Vault watcher disabled in config")
        return
    interval = float(cfg.vault_watcher.poll_interval_seconds)
    logger.info("Starting vault watcher | interval=%ss prefix=Pawn/Tasks", interval)
    active = registry or SallmSessionRegistry()
    try:
        while True:
            try:
                stats = await run_vault_watcher_tick(cfg, registry=active)
                if any(stats.values()):
                    logger.info("vault_watcher tick %s", stats)
            except Exception as exc:
                logger.error("vault_watcher tick failed: %s", exc, exc_info=True)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("Vault watcher cancelled — shutting down cleanly")
        raise
