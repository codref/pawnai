"""Vault task lifecycle: HTTP accept, execute, approve."""

from __future__ import annotations

import asyncio
import logging
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from pawn_agent.core.agent_runner import run_agent_turn
from pawn_agent.core.sallm_registry import SallmSessionRegistry
from pawn_agent.tools.push_queue_message import push_queue_message_impl
from pawn_agent.utils.db import (
    claim_vault_task,
    get_vault_task,
    update_vault_task,
    upsert_vault_task,
)
from pawn_core.vault import VaultNotFound, resolve_path_template
from pawn_core.vault_config import vault_store_from_config
from pawn_server.core.vault_protocol import (
    build_agent_prompt,
    build_obsidian_open_url,
    conversation_id_for_note,
    extract_wiki_links,
    instruction_hash,
    parse_task_note,
    render_task_note,
    wiki_link_to_vault_key,
)

logger = logging.getLogger(__name__)

# In-flight HTTP continuations (task_id -> asyncio.Task)
_background_tasks: dict[str, asyncio.Task[None]] = {}


@dataclass(frozen=True)
class VaultTaskResult:
    """Outcome returned to the Obsidian plugin."""

    task_id: str
    status: str
    result: str
    agent_run_id: Optional[str] = None
    error_code: Optional[str] = None


def task_key_for(cfg: Any, *, task_id: str, note_path: str | None = None) -> str:
    """Resolve the vault object key for a task note (always under Tasks/)."""
    del note_path  # linked source note — not the task file path
    template = getattr(cfg.vault, "task_path_template", "{agent_root}/Tasks/{id}.md")
    return resolve_path_template(
        template,
        agent_root=cfg.vault.agent_root,
        task_id=task_id,
    )


def _effective_conversation(
    conversation: str | None,
    note_path: str | None,
    task_key: str,
) -> str:
    if conversation and conversation.strip():
        return conversation.strip()
    if note_path:
        return conversation_id_for_note(note_path)
    return conversation_id_for_note(task_key)


async def _gather_context(
    cfg: Any,
    store: Any,
    *,
    parsed: dict[str, Any],
    task_key: str,
    note_path: str | None,
) -> str:
    instruction = parsed.get("instruction") or ""
    task_context = parsed.get("context") or ""
    note_excerpt = ""
    if note_path:
        key = wiki_link_to_vault_key(note_path)
        try:
            note_excerpt = await asyncio.to_thread(store.read, key)
        except VaultNotFound:
            note_excerpt = ""
        except Exception as exc:
            logger.debug("Could not read linked note %s: %s", key, exc)
    link_text = "\n".join([instruction, task_context, str(parsed.get("result") or "")])
    links = extract_wiki_links(link_text)
    linked: list[tuple[str, str]] = []
    for link in links[:8]:
        key = wiki_link_to_vault_key(link)
        try:
            body = await asyncio.to_thread(store.read, key)
            linked.append((link, body[:6000]))
        except Exception:
            linked.append((link, ""))
    return build_agent_prompt(
        task_id=str(parsed.get("id") or ""),
        instruction=instruction,
        task_context=task_context,
        note_excerpt=note_excerpt,
        linked_excerpts=linked,
        task_key=task_key,
        note_path=note_path,
    )


async def _notify_matrix(cfg: Any, *, task_id: str, task_key: str, title: str) -> Optional[str]:
    target = cfg.vault_watcher.matrix_target or "matrix"
    vault_name = getattr(cfg.vault, "obsidian_vault_name", "") or ""
    link = build_obsidian_open_url(vault_name, task_key)
    payload = {
        "text": (
            f"Pawn vault result ready for review.\n"
            f"{title[:80]}\n"
            f"Open: {link}\n"
            f"Task: {task_id}"
        ),
        "task_id": task_id,
        "task_key": task_key,
        "kind": "vault_review",
    }
    try:
        receipt = await push_queue_message_impl(
            cfg, target=target, command="notify", payload=payload
        )
        if receipt.startswith("Error"):
            logger.warning("Matrix queue_push failed: %s", receipt)
            return None
        return receipt
    except Exception as exc:
        logger.warning("Matrix notify failed: %s", exc)
        return None


async def execute_vault_task(
    cfg: Any,
    task_id: str,
    *,
    registry: SallmSessionRegistry,
    store: Any,
    write_result_to_vault: bool,
) -> VaultTaskResult:
    """Claimed task: run agent and optionally write ## Result to the vault."""
    row = get_vault_task(cfg.db_dsn, task_id)
    if row is None:
        return VaultTaskResult(
            task_id=task_id,
            status="blocked",
            result="",
            error_code="not_found",
        )

    task_key = row.key
    update_vault_task(cfg.db_dsn, task_id, status="running")

    parsed: dict[str, Any] = {
        "id": task_id,
        "instruction": row.instruction_text or "",
        "context": "",
        "result": "",
    }
    note_path = row.note_path
    etag: Optional[str] = row.etag

    if write_result_to_vault or not (row.instruction_text or "").strip():
        try:
            body = await asyncio.to_thread(store.read, task_key)
            parsed = parse_task_note(body)
            note_path = parsed.get("note_path") or note_path
            st = await asyncio.to_thread(store.stat, task_key)
            if st is not None:
                etag = st.etag
        except Exception as exc:
            logger.warning("Could not read task note %s: %s", task_key, exc)

    if write_result_to_vault:
        try:
            body = await asyncio.to_thread(store.read, task_key)
            parsed = parse_task_note(body)
            meta = dict(parsed.get("meta") or {})
            meta["status"] = "running"
            updated = render_task_note(
                task_id=task_id,
                status="running",
                instruction=parsed.get("instruction") or "",
                context=parsed.get("context") or "",
                result=parsed.get("result") or "",
                conversation=row.conversation_id,
                note_path=note_path,
                approved=bool(parsed.get("approved")),
                extra_meta={
                    k: v for k, v in meta.items() if k not in {"pawn", "id", "status", "approved"}
                },
            )
            st = await asyncio.to_thread(store.write, task_key, updated)
            etag = st.etag
        except Exception as exc:
            logger.warning("Could not mark task running in vault: %s", exc)

    prompt = await _gather_context(
        cfg,
        store,
        parsed=parsed,
        task_key=task_key,
        note_path=note_path,
    )
    try:
        result = await run_agent_turn(
            cfg=cfg,
            registry=registry,
            prompt=prompt,
            session_id=row.conversation_id,
            source="vault",
            command="vault_run",
        )
    except Exception as exc:
        logger.error("vault_run %s failed: %s", task_id, exc, exc_info=True)
        update_vault_task(
            cfg.db_dsn,
            task_id,
            status="blocked",
            error_code="agent_failed",
            etag=etag,
        )
        if write_result_to_vault:
            try:
                body = await asyncio.to_thread(store.read, task_key)
                parsed = parse_task_note(body)
                updated = render_task_note(
                    task_id=task_id,
                    status="blocked",
                    instruction=parsed.get("instruction") or "",
                    context=parsed.get("context") or "",
                    result=str(exc)[:4000],
                    conversation=row.conversation_id,
                    note_path=note_path,
                    approved=False,
                )
                st = await asyncio.to_thread(store.write, task_key, updated)
                etag = st.etag
            except Exception:
                pass
        return VaultTaskResult(
            task_id=task_id,
            status="blocked",
            result="",
            error_code="agent_failed",
        )

    response_text = result.response or ""
    notify_id = await _notify_matrix(
        cfg,
        task_id=task_id,
        task_key=task_key,
        title=row.instruction_text or parsed.get("instruction") or "Vault task",
    )
    update_vault_task(
        cfg.db_dsn,
        task_id,
        status="review",
        agent_run_id=result.run_id,
        matrix_notify_id=notify_id,
        etag=etag,
    )

    if write_result_to_vault:
        try:
            body = await asyncio.to_thread(store.read, task_key)
            parsed = parse_task_note(body)
            updated = render_task_note(
                task_id=task_id,
                status="review",
                instruction=parsed.get("instruction") or "",
                context=parsed.get("context") or "",
                result=response_text,
                conversation=row.conversation_id,
                note_path=note_path,
                approved=False,
            )
            st = await asyncio.to_thread(store.write, task_key, updated)
            update_vault_task(cfg.db_dsn, task_id, etag=st.etag)
        except Exception as exc:
            logger.error("Failed writing vault result for %s: %s", task_id, exc)

    return VaultTaskResult(
        task_id=task_id,
        status="review",
        result=response_text,
        agent_run_id=result.run_id,
    )


async def accept_vault_task_http(
    cfg: Any,
    *,
    task_id: str,
    instruction: str,
    note_path: str | None,
    context: str | None,
    conversation_id: str | None,
    registry: SallmSessionRegistry,
) -> VaultTaskResult:
    """Fast HTTP path: DB + agent run; plugin keeps the task file locally."""
    key = task_key_for(cfg, task_id=task_id, note_path=note_path)
    conv = _effective_conversation(conversation_id, note_path, key)
    ih = instruction_hash(instruction)
    effective_id = upsert_vault_task(
        cfg.db_dsn,
        task_id=task_id,
        key=key,
        instruction_hash=ih,
        conversation_id=conv,
        instruction_text=instruction,
        note_path=note_path,
        via="http",
        status="queued",
    )
    if not claim_vault_task(cfg.db_dsn, effective_id):
        row = get_vault_task(cfg.db_dsn, effective_id)
        if row and row.status == "review":
            return VaultTaskResult(
                task_id=effective_id,
                status="review",
                result="",
                agent_run_id=row.agent_run_id,
            )
        return VaultTaskResult(
            task_id=effective_id,
            status=row.status if row else "blocked",
            result="",
            error_code="not_claimable",
        )

    store = vault_store_from_config(cfg)
    # Inject context into a synthetic parse for gather (file not on S3 fast path).
    parsed = {
        "id": effective_id,
        "instruction": instruction,
        "context": context or "",
        "result": "",
        "note_path": note_path,
    }
    update_vault_task(cfg.db_dsn, effective_id, status="running")
    prompt = await _gather_context(
        cfg,
        store,
        parsed=parsed,
        task_key=key,
        note_path=note_path,
    )
    try:
        result = await run_agent_turn(
            cfg=cfg,
            registry=registry,
            prompt=prompt,
            session_id=conv,
            source="vault",
            command="vault_run",
        )
    except Exception as exc:
        logger.error("vault HTTP run %s failed: %s", effective_id, exc, exc_info=True)
        update_vault_task(
            cfg.db_dsn,
            effective_id,
            status="blocked",
            error_code="agent_failed",
        )
        return VaultTaskResult(
            task_id=effective_id,
            status="blocked",
            result="",
            error_code="agent_failed",
        )

    response_text = result.response or ""
    notify_id = await _notify_matrix(
        cfg,
        task_id=effective_id,
        task_key=key,
        title=instruction,
    )
    update_vault_task(
        cfg.db_dsn,
        effective_id,
        status="review",
        agent_run_id=result.run_id,
        matrix_notify_id=notify_id,
    )
    return VaultTaskResult(
        task_id=effective_id,
        status="review",
        result=response_text,
        agent_run_id=result.run_id,
    )


async def approve_vault_task(
    cfg: Any,
    task_id: str,
    *,
    registry: SallmSessionRegistry,
    store: Any | None = None,
    result_text: str | None = None,
) -> VaultTaskResult:
    """Index approved result into sallm memory and mark task done."""
    row = get_vault_task(cfg.db_dsn, task_id)
    if row is None:
        return VaultTaskResult(
            task_id=task_id,
            status="blocked",
            result="",
            error_code="not_found",
        )
    if row.status not in {"review", "done"}:
        return VaultTaskResult(
            task_id=task_id,
            status=row.status,
            result="",
            error_code="not_ready",
        )

    if row.indexed_at is not None:
        if row.status != "done":
            update_vault_task(cfg.db_dsn, task_id, status="done")
        return VaultTaskResult(
            task_id=task_id, status="done", result="", agent_run_id=row.agent_run_id
        )

    text = result_text or ""
    task_key = row.key
    if not text.strip() and store is not None:
        try:
            body = await asyncio.to_thread(store.read, task_key)
            parsed = parse_task_note(body)
            text = parsed.get("result") or ""
        except Exception as exc:
            logger.warning("Could not read result from vault for %s: %s", task_id, exc)

    summary = (
        "Approved vault task result.\n"
        f"task_id={task_id}\n"
        f"task_key={task_key}\n\n"
        f"Instruction:\n{(row.instruction_text or '')[:1500]}\n\n"
        f"Result:\n{(text or '')[:6000]}"
    )
    try:
        session = await registry.get_or_create(row.conversation_id, cfg, cfg.db_dsn)
        await asyncio.to_thread(
            session._agent.remember,  # noqa: SLF001 — intentional index path
            summary,
            source=f"vault:{task_id}",
            index_raw=False,
        )
    except Exception as exc:
        logger.error("remember() failed for vault task %s: %s", task_id, exc, exc_info=True)
        update_vault_task(cfg.db_dsn, task_id, error_code="index_failed")
        return VaultTaskResult(
            task_id=task_id,
            status="review",
            result="",
            error_code="index_failed",
        )

    now = datetime.now(timezone.utc)
    update_vault_task(
        cfg.db_dsn,
        task_id,
        status="done",
        indexed_at=now,
    )

    if store is not None:
        try:
            body = await asyncio.to_thread(store.read, task_key)
            parsed = parse_task_note(body)
            updated = render_task_note(
                task_id=task_id,
                status="done",
                instruction=parsed.get("instruction") or "",
                context=parsed.get("context") or "",
                result=text or parsed.get("result") or "",
                conversation=row.conversation_id,
                note_path=row.note_path,
                approved=True,
            )
            await asyncio.to_thread(store.write, task_key, updated)
        except Exception as exc:
            logger.warning("Could not update vault task note on approve: %s", exc)

    return VaultTaskResult(
        task_id=task_id,
        status="done",
        result=text,
        agent_run_id=row.agent_run_id,
    )


async def write_result_to_vault_note(
    cfg: Any,
    task_id: str,
    *,
    result_text: str,
) -> None:
    """Write ## Result and status=review onto the task note in S3."""
    row = get_vault_task(cfg.db_dsn, task_id)
    if row is None:
        return
    store = vault_store_from_config(cfg)
    task_key = row.key
    try:
        body = await asyncio.to_thread(store.read, task_key)
        parsed = parse_task_note(body)
        updated = render_task_note(
            task_id=task_id,
            status="review",
            instruction=parsed.get("instruction") or row.instruction_text or "",
            context=parsed.get("context") or "",
            result=result_text,
            conversation=row.conversation_id,
            note_path=row.note_path,
            approved=False,
        )
        st = await asyncio.to_thread(store.write, task_key, updated)
        update_vault_task(cfg.db_dsn, task_id, etag=st.etag)
    except Exception as exc:
        logger.error("Failed slow-path vault write for %s: %s", task_id, exc)


async def run_http_vault_task_with_optional_vault_write(
    cfg: Any,
    *,
    task_id: str,
    instruction: str,
    note_path: str | None,
    context: str | None,
    conversation_id: str | None,
    registry: SallmSessionRegistry,
    write_result_to_vault: asyncio.Event,
) -> VaultTaskResult:
    """HTTP runner; writes the task note when *write_result_to_vault* is set."""
    result = await accept_vault_task_http(
        cfg,
        task_id=task_id,
        instruction=instruction,
        note_path=note_path,
        context=context,
        conversation_id=conversation_id,
        registry=registry,
    )
    if write_result_to_vault.is_set() and result.status == "review":
        await write_result_to_vault_note(cfg, task_id, result_text=result.result)
    return result


def track_background_task(task_id: str, task: asyncio.Task[Any]) -> None:
    _background_tasks[task_id] = task

    def _done(_: asyncio.Task[Any]) -> None:
        _background_tasks.pop(task_id, None)

    task.add_done_callback(_done)


def new_task_id() -> str:
    return str(uuid.uuid4())
