"""Outbound coworker notifications: vault is always written by the caller.

Matrix uses the existing queue producer. ntfy is optional.
"""

from __future__ import annotations

import logging
import urllib.error
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)


async def notify(
    cfg: Any,
    *,
    kind: str,
    text: str,
    link: str = "",
    item_id: str = "",
) -> None:
    """Deliver *text*. Failures are recorded and do not raise."""
    from pawn_agent.core.coworker import db as itemdb  # noqa: PLC0415
    from pawn_server.core.vault_protocol import build_obsidian_open_url  # noqa: PLC0415

    vault_name = getattr(cfg.vault, "obsidian_vault_name", "") or ""
    open_url = build_obsidian_open_url(vault_name, link) if link and vault_name else ""
    body = text if not open_url else f"{text}\nOpen: {open_url}"
    outcome = "sent"
    error: Optional[str] = None
    try:
        await _matrix(cfg, kind=kind, text=body, item_id=item_id, link=link)
    except Exception as exc:
        outcome = "matrix_failed"
        error = str(exc)
        logger.warning("coworker matrix notify failed: %s", exc)
    ntfy_error = _ntfy(cfg, text=body, link=open_url or link)
    if ntfy_error:
        error = (error + "; " if error else "") + ntfy_error
        if outcome == "sent":
            outcome = "ntfy_failed"
    try:
        itemdb.record_decision(
            cfg.db_dsn,
            event_kind="notify",
            policy_decision="allow",
            event_id=item_id or None,
            proposed_action=kind,
            outcome=outcome,
            error=error,
        )
    except Exception as exc:
        logger.debug("notify audit skipped: %s", exc)


async def _matrix(cfg: Any, *, kind: str, text: str, item_id: str, link: str) -> None:
    target = getattr(cfg.coworker, "matrix_target", None) or "matrix"
    if not getattr(cfg, "queue_producers", None) or target not in (cfg.queue_producers or {}):
        return
    from pawn_agent.tools.push_queue_message import push_queue_message_impl  # noqa: PLC0415

    receipt = await push_queue_message_impl(
        cfg,
        target=target,
        command="notify",
        payload={"text": text, "kind": kind, "item_id": item_id, "task_key": link},
    )
    if str(receipt).startswith("Error"):
        raise RuntimeError(receipt)


def _ntfy(cfg: Any, *, text: str, link: str) -> Optional[str]:
    notify_cfg = getattr(cfg.coworker, "notify", None)
    base = (getattr(notify_cfg, "ntfy_url", "") or "").rstrip("/")
    topic = getattr(notify_cfg, "topic", "") or ""
    if not base or not topic:
        return None
    url = f"{base}/{topic}"
    headers = {"Title": "Pawn", "Content-Type": "text/plain"}
    token = getattr(notify_cfg, "token", "") or ""
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if link:
        headers["Click"] = link
    request = urllib.request.Request(url, data=text.encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            if response.status >= 300:
                return f"ntfy status {response.status}"
    except urllib.error.URLError as exc:
        return f"ntfy: {exc}"
    return None
