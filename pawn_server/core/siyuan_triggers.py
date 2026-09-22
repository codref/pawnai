"""HTTP-facing SiYuan @pawn trigger: enqueue + claim from a block id."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from pawn_agent.core.sallm_registry import SallmSessionRegistry
from pawn_agent.core.siyuan_protocol import (
    ATTR_REQUEST_ID,
    ATTR_SOURCE_HASH,
    ATTR_STATUS,
    STATUSES_NO_RETRIGGER,
    client_from_agent_config,
    conversation_id_for_root,
    instruction_hash,
    new_request_id,
    resolve_notebook_allowlist,
    resolve_pawn_trigger,
)
from pawn_agent.utils.db import (
    claim_siyuan_agent_request,
    get_siyuan_agent_request,
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
    """Resolve *block_id* to a pawn callout, upsert, and start execution if needed.

    Returns immediately after scheduling ``execute_claimed_request`` when a new
    run is started. Idempotent for the same trigger + instruction hash while the
    request is still queued/claimed/running/review.
    """
    mention = cfg.siyuan_watcher.mention_token
    active_client = client or client_from_agent_config(cfg)
    # Plugin Send is explicit intent — do not require an @pawn token in the
    # block. Watcher discovery still filters on mention_token via SQL.
    resolved = resolve_pawn_trigger(
        active_client,
        block_id,
        mention_token=mention,
        require_mention=False,
    )
    if resolved is None:
        raise SiyuanTriggerError(
            "No TIP callout or non-empty block found for this block_id",
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
        raise SiyuanTriggerError("Callout instruction is empty", status_code=400)

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

    import asyncio

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
