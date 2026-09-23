"""Pull-only SiYuan @pawn watcher for pawn-server.

Discovers ``@pawn`` blocks via SQL, claims durable Postgres rows, runs the
agent (in-process), appends review drafts, Matrix-notifies, and indexes
approvals into sallm memory.
"""

from __future__ import annotations

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import Any, Optional

from pawn_agent.core.agent_runner import run_agent_turn
from pawn_agent.core.sallm_registry import SallmSessionRegistry
from pawn_agent.core.siyuan_protocol import (
    ATTR_OUTPUT_ID,
    ATTR_REQUEST_ID,
    ATTR_SOURCE_HASH,
    ATTR_STATUS,
    STATUSES_SKIP_REDISCOVER,
    InstructionSettleTracker,
    approval_checked,
    build_agent_prompt,
    build_discovery_sql,
    build_result_markdown,
    client_from_agent_config,
    conversation_id_for_root,
    extract_block_refs,
    instruction_hash,
    new_request_id,
    parse_discovered_rows,
    resolve_notebook_allowlist,
    strip_mention_from_instruction,
)
from pawn_agent.tools.push_queue_message import push_queue_message_impl
from pawn_agent.utils.db import (
    claim_siyuan_agent_request,
    get_siyuan_agent_request,
    get_siyuan_poll_watermark,
    list_siyuan_agent_requests,
    set_siyuan_poll_watermark,
    update_siyuan_agent_request,
    upsert_siyuan_agent_request,
)
from pawn_agent.utils.siyuan import build_siyuan_block_url

logger = logging.getLogger(__name__)

_OUTPUT_ID_RE = re.compile(r"block_id=([0-9a-zA-Z\-]+)")
# Process-local debounce so mid-edit autosaves are not claimed.
_SETTLE = InstructionSettleTracker()


def _set_trigger_attrs(client: Any, block_id: str, attrs: dict[str, str]) -> None:
    try:
        client.set_block_attrs(block_id, attrs)
    except Exception as exc:
        logger.warning("Failed to set attrs on %s: %s", block_id, exc)


def _settle_seconds(cfg: Any) -> float:
    return float(getattr(cfg.siyuan_watcher, "settle_seconds", 45.0) or 0.0)


def discover_and_enqueue(cfg: Any, client: Any) -> int:
    """SQL-scan for new @pawn blocks and upsert queued requests. Returns count.

    Instructions are only enqueued after their text hash has been unchanged for
    ``siyuan_watcher.settle_seconds`` (default 45s), so SiYuan autosave while
    typing does not trigger a premature agent run.
    """
    notebooks = resolve_notebook_allowlist(cfg)
    if not notebooks:
        logger.debug("siyuan_watcher: no notebook allowlist / siyuan.notebook — skip")
        return 0
    watcher = cfg.siyuan_watcher
    watermark = get_siyuan_poll_watermark(cfg.db_dsn)
    stmt = build_discovery_sql(
        notebooks,
        mention_token=watcher.mention_token,
        watermark=watermark,
        limit=50,
    )
    try:
        rows = client.query_sql(stmt)
    except Exception as exc:
        logger.error("siyuan_watcher SQL discovery failed: %s", exc, exc_info=True)
        return 0

    discovered = parse_discovered_rows(rows, mention_token=watcher.mention_token)
    created = 0
    max_updated = watermark
    for block in discovered:
        if block.updated and block.updated > max_updated:
            max_updated = block.updated

    # First run: advance watermark without claiming historical @pawn blocks.
    if not watermark:
        if max_updated:
            set_siyuan_poll_watermark(cfg.db_dsn, max_updated)
            logger.info(
                "siyuan_watcher bootstrap watermark=%s (skipped %d historical hits)",
                max_updated,
                len(discovered),
            )
        return 0

    settle = _settle_seconds(cfg)
    # Only advance watermark past blocks we finished deciding on (settled).
    max_safe_watermark = watermark

    for block in discovered:
        attrs = client.get_block_attrs(block.block_id)
        status = (attrs.get(ATTR_STATUS) or "").strip().lower()
        text = block.instruction_text
        ih = instruction_hash(text)
        if status in STATUSES_SKIP_REDISCOVER and attrs.get(ATTR_SOURCE_HASH) == ih:
            _SETTLE.forget(block.block_id)
            if block.updated and block.updated > max_safe_watermark:
                max_safe_watermark = block.updated
            continue

        # Already queued with this exact text — nothing to do.
        if status == "queued" and attrs.get(ATTR_SOURCE_HASH) == ih:
            _SETTLE.forget(block.block_id)
            if block.updated and block.updated > max_safe_watermark:
                max_safe_watermark = block.updated
            continue

        if not _SETTLE.is_settled(block.block_id, ih, settle_seconds=settle):
            logger.debug(
                "siyuan_watcher waiting for settle block=%s settle=%ss",
                block.block_id,
                settle,
            )
            # Do not advance watermark past unsettled blocks — keep re-polling.
            continue

        # Prefer a fresh UUID when the attr id belonged to a prior hash (edit race).
        prev_hash = (attrs.get(ATTR_SOURCE_HASH) or "").strip()
        prev_rid = (attrs.get(ATTR_REQUEST_ID) or "").strip()
        if prev_rid and prev_hash and prev_hash == ih:
            request_id = prev_rid
        else:
            request_id = new_request_id()
        conv = conversation_id_for_root(block.root_id)
        effective_id = upsert_siyuan_agent_request(
            cfg.db_dsn,
            request_id=request_id,
            trigger_block_id=block.block_id,
            parent_block_id=block.parent_id or block.block_id,
            root_id=block.root_id,
            notebook_id=block.notebook_id,
            instruction_hash=ih,
            conversation_id=conv,
            instruction_text=text,
            source_updated=block.updated,
            status="queued",
        )
        _set_trigger_attrs(
            client,
            block.block_id,
            {
                ATTR_STATUS: "queued",
                ATTR_REQUEST_ID: effective_id,
                ATTR_SOURCE_HASH: ih,
            },
        )
        _SETTLE.forget(block.block_id)
        if block.updated and block.updated > max_safe_watermark:
            max_safe_watermark = block.updated
        created += 1

    if max_safe_watermark and max_safe_watermark != watermark:
        set_siyuan_poll_watermark(cfg.db_dsn, max_safe_watermark)
    return created


def _gather_context(cfg: Any, client: Any, request: Any) -> str:
    instruction = request.instruction_text or ""
    mention = cfg.siyuan_watcher.mention_token
    clean = strip_mention_from_instruction(instruction, mention)
    parent_excerpt = ""
    try:
        parent_excerpt = client.get_block_kramdown(request.parent_block_id) or ""
    except Exception:
        pass
    refs = extract_block_refs(instruction)
    ref_excerpts: list[tuple[str, str]] = []
    max_refs = max(0, int(cfg.siyuan_watcher.max_context_blocks) - 2)
    for rid in refs[:max_refs]:
        try:
            ref_excerpts.append((rid, client.get_block_kramdown(rid) or ""))
        except Exception:
            ref_excerpts.append((rid, ""))
    return build_agent_prompt(
        request_id=request.id,
        instruction=clean,
        parent_excerpt=parent_excerpt,
        ref_excerpts=ref_excerpts,
        parent_block_id=request.parent_block_id,
        trigger_block_id=request.trigger_block_id,
    )


async def _notify_matrix(cfg: Any, request: Any, output_block_id: str) -> Optional[str]:
    target = cfg.siyuan_watcher.matrix_target or "matrix"
    title = (request.instruction_text or "SiYuan @pawn")[:80]
    link = build_siyuan_block_url(output_block_id)
    payload = {
        "text": (
            f"Pawn result ready for review.\n"
            f"{title}\n"
            f"Open: {link}\n"
            f"Request: {request.id}"
        ),
        "request_id": request.id,
        "output_block_id": output_block_id,
        "kind": "siyuan_review",
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


async def execute_claimed_request(
    cfg: Any,
    request_id: str,
    *,
    registry: SallmSessionRegistry,
    client: Any,
) -> None:
    """Run one claimed request through the agent and mark review."""
    request = get_siyuan_agent_request(cfg.db_dsn, request_id)
    if request is None:
        return
    update_siyuan_agent_request(cfg.db_dsn, request_id, status="running")
    # The prompt (or legacy callout) is owned by the SiYuan plugin — do not
    # rewrite the trigger block here.
    _set_trigger_attrs(
        client,
        request.trigger_block_id,
        {
            ATTR_STATUS: "running",
            ATTR_REQUEST_ID: request.id,
            ATTR_SOURCE_HASH: request.instruction_hash,
        },
    )
    prompt = await asyncio.to_thread(_gather_context, cfg, client, request)
    try:
        result = await run_agent_turn(
            cfg=cfg,
            registry=registry,
            prompt=prompt,
            session_id=request.conversation_id,
            source="siyuan",
            command="siyuan_run",
        )
    except Exception as exc:
        logger.error("siyuan_run %s failed: %s", request_id, exc, exc_info=True)
        update_siyuan_agent_request(
            cfg.db_dsn, request_id, status="blocked", error_code="agent_failed"
        )
        _set_trigger_attrs(client, request.trigger_block_id, {ATTR_STATUS: "blocked"})
        return

    output_id = ""
    match = _OUTPUT_ID_RE.search(result.response or "")
    if match:
        output_id = match.group(1)
    # Coordinator fallback: if the model did not append, write the reply.
    if not output_id:
        try:
            body = build_result_markdown(
                result.response or "_Agent produced no structured append._",
                request_id=request.id,
            )
            output_id = await asyncio.to_thread(client.append_block, request.parent_block_id, body)
        except Exception as exc:
            logger.error("Fallback append failed for %s: %s", request_id, exc)
            update_siyuan_agent_request(
                cfg.db_dsn,
                request_id,
                status="blocked",
                error_code="append_failed",
                agent_run_id=result.run_id,
            )
            return

    _set_trigger_attrs(
        client,
        request.trigger_block_id,
        {
            ATTR_STATUS: "review",
            ATTR_REQUEST_ID: request.id,
            ATTR_OUTPUT_ID: output_id,
        },
    )
    notify_id = await _notify_matrix(cfg, request, output_id)
    update_siyuan_agent_request(
        cfg.db_dsn,
        request_id,
        status="review",
        output_block_id=output_id,
        agent_run_id=result.run_id,
        matrix_notify_id=notify_id,
    )


async def process_approvals(
    cfg: Any,
    *,
    registry: SallmSessionRegistry,
    client: Any,
) -> int:
    """Poll review requests for Approve checkbox; mark done + remember."""
    rows = list_siyuan_agent_requests(cfg.db_dsn, status="review", limit=30)
    done_count = 0
    for request in rows:
        output_id = request.output_block_id
        if not output_id:
            continue
        try:
            kramdown = await asyncio.to_thread(client.get_block_kramdown, output_id)
        except Exception as exc:
            logger.warning("Failed reading output %s: %s", output_id, exc)
            continue
        attrs = await asyncio.to_thread(client.get_block_attrs, request.trigger_block_id)
        status_attr = (attrs.get(ATTR_STATUS) or "").strip().lower()
        approved = approval_checked(kramdown) or status_attr == "done"
        if not approved:
            continue
        if request.indexed_at is not None:
            if status_attr != "done":
                _set_trigger_attrs(client, request.trigger_block_id, {ATTR_STATUS: "done"})
            continue

        summary = (
            f"Approved SiYuan @pawn result.\n"
            f"request_id={request.id}\n"
            f"trigger={request.trigger_block_id}\n"
            f"output={output_id}\n\n"
            f"Instruction:\n{(request.instruction_text or '')[:1500]}\n\n"
            f"Result:\n{(kramdown or '')[:6000]}"
        )
        try:
            session = await registry.get_or_create(request.conversation_id, cfg, cfg.db_dsn)
            await asyncio.to_thread(
                session._agent.remember,  # noqa: SLF001 — intentional index path
                summary,
                source=f"siyuan:{request.id}",
                index_raw=False,
            )
        except Exception as exc:
            logger.error("remember() failed for %s: %s", request.id, exc, exc_info=True)
            update_siyuan_agent_request(cfg.db_dsn, request.id, error_code="index_failed")
            continue

        now = datetime.now(timezone.utc)
        update_siyuan_agent_request(cfg.db_dsn, request.id, status="done", indexed_at=now)
        _set_trigger_attrs(client, request.trigger_block_id, {ATTR_STATUS: "done"})
        done_count += 1
    return done_count


async def run_siyuan_watcher_tick(
    cfg: Any,
    *,
    registry: Optional[SallmSessionRegistry] = None,
) -> dict[str, int]:
    """One watcher tick: discover (optional), claim/execute, process approvals."""
    active = registry or SallmSessionRegistry()
    client = client_from_agent_config(cfg)
    discover = bool(getattr(cfg.siyuan_watcher, "discover_mentions", False))
    discovered = 0
    if discover:
        discovered = await asyncio.to_thread(discover_and_enqueue, cfg, client)

    queued = list_siyuan_agent_requests(cfg.db_dsn, status="queued", limit=20)
    claimed = 0
    max_claims = max(1, int(cfg.siyuan_watcher.max_claims_per_tick))
    for request in queued:
        if claimed >= max_claims:
            break
        if not claim_siyuan_agent_request(cfg.db_dsn, request.id):
            continue
        _set_trigger_attrs(
            client,
            request.trigger_block_id,
            {ATTR_STATUS: "claimed", ATTR_REQUEST_ID: request.id},
        )
        await execute_claimed_request(cfg, request.id, registry=active, client=client)
        claimed += 1

    approved = await process_approvals(cfg, registry=active, client=client)
    return {"discovered": discovered, "claimed": claimed, "approved": approved}


async def start_siyuan_watcher(
    cfg: Any,
    *,
    registry: Optional[SallmSessionRegistry] = None,
) -> None:
    """Run the SiYuan @pawn watcher loop until cancelled."""
    interval = float(cfg.siyuan_watcher.poll_interval_seconds)
    logger.info(
        "Starting SiYuan watcher | interval=%ss mention=%s discover=%s",
        interval,
        cfg.siyuan_watcher.mention_token,
        bool(getattr(cfg.siyuan_watcher, "discover_mentions", False)),
    )
    active = registry or SallmSessionRegistry()
    try:
        while True:
            try:
                stats = await run_siyuan_watcher_tick(cfg, registry=active)
                if any(stats.values()):
                    logger.info("siyuan_watcher tick %s", stats)
            except Exception as exc:
                logger.error("siyuan_watcher tick failed: %s", exc, exc_info=True)
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        logger.info("SiYuan watcher cancelled — shutting down cleanly")
        raise
