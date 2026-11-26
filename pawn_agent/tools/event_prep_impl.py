"""Write ``## Prep`` on a ``pawn: event`` note only."""

from __future__ import annotations

from typing import Any

from pawn_core.vault import VaultNotFound, VaultWriteDenied
from pawn_core.vault_config import vault_store_from_config
from pawn_server.core.planner import (
    find_event_by_id,
    replace_prep_section,
)
from pawn_server.core.vault_events import publish_vault_event


def event_prep_update_impl(
    cfg: Any,
    *,
    event_id: str = "",
    path: str = "",
    content: str,
    job_id: str = "",
    store: Any = None,
) -> str:
    """Replace ``## Prep`` on the target event. Returns a one-line receipt."""
    if store is None:
        store = vault_store_from_config(cfg)
    body = (content or "").strip()
    if not body:
        return "Error: prep content is empty"
    eid = (event_id or "").strip()
    key = (path or "").strip()
    if not eid and not key:
        return "Error: provide --event-id or --path"
    try:
        if eid:
            ev = find_event_by_id(cfg, eid, store=store)
            key = str(ev.get("path") or key)
        if not key:
            return "Error: event path missing"
        existing = store.read(key)
    except VaultNotFound:
        return f"Error: event note not found: {key or eid!r}"
    except Exception as exc:
        return f"Error reading event: {exc}"

    try:
        updated = replace_prep_section(existing, body, prep_job_id=job_id)
        store.write(key, updated)
    except VaultWriteDenied as exc:
        return f"Error: {exc}"
    except ValueError as exc:
        return f"Error: {exc}"
    except Exception as exc:
        return f"Error writing prep: {exc}"

    publish_vault_event([key], source="planner")
    return f"Updated ## Prep on {key}"
