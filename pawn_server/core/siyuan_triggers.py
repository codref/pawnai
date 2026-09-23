"""HTTP-facing SiYuan @pawn trigger: enqueue + claim from a block id."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from pawn_agent.core.sallm_registry import SallmSessionRegistry
from pawn_agent.core.siyuan_protocol import (
    ATTR_REQUEST_ID,
    ATTR_SOURCE_HASH,
    ATTR_STATUS,
    STATUSES_NO_RETRIGGER,
    approval_checked,
    client_from_agent_config,
    conversation_id_for_root,
    fetch_block_row,
    find_nearby_request_id,
    match_request_by_output_position,
    instruction_hash,
    new_request_id,
    resolve_notebook_allowlist,
    resolve_pawn_trigger,
)
from pawn_agent.utils.db import (
    claim_siyuan_agent_request,
    get_siyuan_agent_request,
    list_siyuan_agent_requests,
    update_siyuan_agent_request,
    upsert_siyuan_agent_request,
)
from pawn_server.core.siyuan_watcher import execute_claimed_request

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SiyuanTriggerResult:
    """Outcome of accepting a plugin/API trigger."""

    request_id: str
    status: str
    conversation_id: str
    trigger_block_id: str
    started: bool


@dataclass(frozen=True)
class SiyuanApprovalResult:
    """Outcome of indexing one Approve click."""

    request_id: str
    status: str
    indexed: bool


class SiyuanTriggerError(Exception):
    """User-facing trigger rejection (maps to HTTP 4xx)."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.message = message


def _set_trigger_attrs(client: Any, block_id: str, attrs: dict[str, str]) -> None:
    try:
        client.set_block_attrs(block_id, attrs)
    except Exception as exc:
        logger.warning("Failed to set attrs on %s: %s", block_id, exc)


async def accept_siyuan_trigger(
    cfg: Any,
    block_id: str,
    *,
    registry: SallmSessionRegistry,
    client: Any | None = None,
) -> SiyuanTriggerResult:
    """Resolve *block_id* to a Pawn prompt (or legacy mention), upsert, and start.

    Returns immediately after scheduling ``execute_claimed_request`` when a new
    run is started. Idempotent for the same trigger + instruction hash while the
    request is still queued/claimed/running/review.
    """
    mention = cfg.siyuan_watcher.mention_token
    active_client = client or client_from_agent_config(cfg)
    resolved = resolve_pawn_trigger(active_client, block_id, mention_token=mention)
    if resolved is None:
        raise SiyuanTriggerError(
            "No Pawn prompt (or legacy @pawn block) found for this block_id",
            status_code=404,
        )

    allowlist = resolve_notebook_allowlist(cfg)
    if allowlist and resolved.notebook_id not in allowlist:
        raise SiyuanTriggerError(
            f"Block notebook {resolved.notebook_id!r} is not allowlisted",
            status_code=403,
        )

    text = resolved.instruction_text
    if not text:
        raise SiyuanTriggerError("Instruction is empty", status_code=400)

    ih = instruction_hash(text)
    conv = conversation_id_for_root(resolved.root_id)
    request_id = new_request_id()
    effective_id = upsert_siyuan_agent_request(
        cfg.db_dsn,
        request_id=request_id,
        trigger_block_id=resolved.trigger_block_id,
        parent_block_id=resolved.parent_block_id,
        root_id=resolved.root_id,
        notebook_id=resolved.notebook_id,
        instruction_hash=ih,
        conversation_id=conv,
        instruction_text=text,
        source_updated=resolved.source_updated or None,
        status="queued",
    )
    request = get_siyuan_agent_request(cfg.db_dsn, effective_id)
    if request is None:
        raise SiyuanTriggerError("Failed to persist agent request", status_code=500)

    # Already in-flight or awaiting review for this exact instruction — no re-run.
    if request.status in STATUSES_NO_RETRIGGER and request.status != "queued":
        return SiyuanTriggerResult(
            request_id=request.id,
            status=request.status,
            conversation_id=request.conversation_id,
            trigger_block_id=resolved.trigger_block_id,
            started=False,
        )

    # Terminal (or unexpected) status with the same hash — require an edit to retry.
    if request.status != "queued":
        return SiyuanTriggerResult(
            request_id=request.id,
            status=request.status,
            conversation_id=request.conversation_id,
            trigger_block_id=resolved.trigger_block_id,
            started=False,
        )

    _set_trigger_attrs(
        active_client,
        resolved.trigger_block_id,
        {
            ATTR_STATUS: "queued",
            ATTR_REQUEST_ID: effective_id,
            ATTR_SOURCE_HASH: ih,
        },
    )

    if not claim_siyuan_agent_request(cfg.db_dsn, effective_id):
        # Another worker claimed it between upsert and claim.
        latest = get_siyuan_agent_request(cfg.db_dsn, effective_id)
        status = latest.status if latest else "claimed"
        return SiyuanTriggerResult(
            request_id=effective_id,
            status=status,
            conversation_id=conv,
            trigger_block_id=resolved.trigger_block_id,
            started=False,
        )

    _set_trigger_attrs(
        active_client,
        resolved.trigger_block_id,
        {ATTR_STATUS: "claimed", ATTR_REQUEST_ID: effective_id},
    )

    asyncio.create_task(
        _run_claimed(
            cfg,
            effective_id,
            registry=registry,
            client=active_client,
        ),
        name=f"siyuan-trigger-{effective_id}",
    )
    return SiyuanTriggerResult(
        request_id=effective_id,
        status="claimed",
        conversation_id=conv,
        trigger_block_id=resolved.trigger_block_id,
        started=True,
    )


async def _run_claimed(
    cfg: Any,
    request_id: str,
    *,
    registry: SallmSessionRegistry,
    client: Any,
) -> None:
    try:
        await execute_claimed_request(cfg, request_id, registry=registry, client=client)
    except Exception as exc:
        logger.error(
            "Background siyuan trigger %s failed: %s",
            request_id,
            exc,
            exc_info=True,
        )


def _child_ids(client: Any, root_id: str) -> list[str]:
    getter = getattr(client, "get_child_blocks", None)
    if not root_id or getter is None:
        return []
    try:
        rows = getter(root_id) or []
    except Exception:
        return []
    if not isinstance(rows, list):
        return []
    return [str(row.get("id")) for row in rows if isinstance(row, dict) and row.get("id")]


def _anchor_in_order(client: Any, block_id: str, ordered: list[str]) -> str | None:
    """First id in *ordered* walking from *block_id* through SQL parents."""
    wanted = set(ordered)
    current = block_id
    seen: set[str] = set()
    for _ in range(12):
        if not current or current in seen:
            return None
        if current in wanted:
            return current
        seen.add(current)
        row = fetch_block_row(client, current) or {}
        current = str(row.get("parent_id") or "")
    return None


def _request_id_for_checked_block(
    cfg: Any, client: Any, block_id: str, row: dict[str, Any]
) -> str | None:
    """Match the checkbox to the nearest preceding result on the same document."""
    root_id = str(row.get("root_id") or "")
    ordered = _child_ids(client, root_id)
    if not ordered:
        return None
    anchor = _anchor_in_order(client, block_id, ordered)
    if not anchor:
        return None
    candidates = list_siyuan_agent_requests(
        cfg.db_dsn,
        statuses=["review", "done"],
        root_id=root_id,
        newest_first=True,
        limit=200,
    )
    return match_request_by_output_position(
        ordered,
        anchor,
        [(item.id, item.output_block_id) for item in candidates],
    )


async def accept_siyuan_approval(
    cfg: Any,
    block_id: str,
    *,
    registry: SallmSessionRegistry,
    client: Any | None = None,
) -> SiyuanApprovalResult:
    """Index the Pawn result for a checked Approve checkbox.

    *block_id* is the clicked list item. The request UUID is read from the
    nearby ``_request:`` line. Already-indexed requests return without calling
    ``remember`` again.
    """
    active = client or client_from_agent_config(cfg)
    row = fetch_block_row(active, block_id)
    if row is None:
        raise SiyuanTriggerError("Block not found", status_code=404)

    notebook = str(row.get("box") or "")
    allowlist = resolve_notebook_allowlist(cfg)
    if allowlist and notebook not in allowlist:
        raise SiyuanTriggerError(
            f"Block notebook {notebook!r} is not allowlisted",
            status_code=403,
        )

    try:
        kramdown = await asyncio.to_thread(active.get_block_kramdown, block_id)
    except Exception as exc:
        raise SiyuanTriggerError(
            f"Could not read block: {exc}",
            status_code=502,
        ) from exc
    markdown = str(row.get("markdown") or "")
    if not (approval_checked(kramdown or "") or approval_checked(markdown)):
        logger.warning(
            "Approve checkbox not checked on %s kramdown=%r markdown=%r",
            block_id,
            (kramdown or "")[:240],
            markdown[:240],
        )
        raise SiyuanTriggerError("Approve checkbox is not checked", status_code=409)

    request_id = _request_id_for_checked_block(cfg, active, block_id, row)
    if not request_id:
        request_id = find_nearby_request_id(active, block_id)
    if not request_id:
        raise SiyuanTriggerError(
            "No Pawn result request id near this checkbox",
            status_code=404,
        )
    request = get_siyuan_agent_request(cfg.db_dsn, request_id)
    if request is None:
        raise SiyuanTriggerError("Pawn request not found", status_code=404)
    if request.status not in {"review", "done"}:
        raise SiyuanTriggerError(
            f"Request is {request.status}, not ready to index",
            status_code=409,
        )

    if request.indexed_at is not None:
        _set_trigger_attrs(active, request.trigger_block_id, {ATTR_STATUS: "done"})
        if request.status != "done":
            update_siyuan_agent_request(cfg.db_dsn, request.id, status="done")
        return SiyuanApprovalResult(request_id=request.id, status="done", indexed=False)

    result_text = ""
    output_id = request.output_block_id or ""
    if output_id:
        try:
            result_text = await asyncio.to_thread(active.get_block_kramdown, output_id)
        except Exception as exc:
            logger.warning("Failed reading output %s: %s", output_id, exc)
            result_text = kramdown or ""
    else:
        result_text = kramdown or ""

    summary = (
        "Approved SiYuan @pawn result.\n"
        f"request_id={request.id}\n"
        f"trigger={request.trigger_block_id}\n"
        f"output={output_id}\n\n"
        f"Instruction:\n{(request.instruction_text or '')[:1500]}\n\n"
        f"Result:\n{(result_text or '')[:6000]}"
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
        raise SiyuanTriggerError(
            f"Could not index into Pawn memory: {exc}",
            status_code=500,
        ) from exc

    update_siyuan_agent_request(
        cfg.db_dsn,
        request.id,
        status="done",
        indexed_at=datetime.now(timezone.utc),
    )
    _set_trigger_attrs(active, request.trigger_block_id, {ATTR_STATUS: "done"})
    return SiyuanApprovalResult(request_id=request.id, status="done", indexed=True)
