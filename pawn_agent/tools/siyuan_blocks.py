"""SiYuan block read/append helpers for agent CliTools (append-only path)."""

from __future__ import annotations

from typing import Optional

from pawn_agent.core.siyuan_protocol import (
    ATTR_OUTPUT_ID,
    ATTR_REQUEST_ID,
    ATTR_RUN_ID,
    ATTR_SOURCE_HASH,
    ATTR_STATUS,
    STATUSES_ACTIVE,
    STATUSES_TERMINAL,
    build_result_markdown,
    client_from_agent_config,
    extract_block_refs,
)
from pawn_agent.utils.config import AgentConfig
from pawn_agent.utils.siyuan import build_siyuan_block_url

_ALLOWED_STATUS = STATUSES_ACTIVE | STATUSES_TERMINAL | frozenset({"queued"})
_ALLOWED_ATTRS = frozenset(
    {ATTR_STATUS, ATTR_REQUEST_ID, ATTR_OUTPUT_ID, ATTR_SOURCE_HASH, ATTR_RUN_ID}
)


def siyuan_read_impl(
    cfg: AgentConfig,
    *,
    block_id: str,
    include_children: bool = False,
    include_attrs: bool = False,
    resolve_refs: bool = False,
    max_ref_depth: int = 1,
    max_blocks: int = 40,
) -> str:
    """Read kramdown (and optional children/attrs/refs) for *block_id*."""
    if not block_id:
        return "Error: --block-id is required."
    client = client_from_agent_config(cfg)
    parts: list[str] = [f"block_id: {block_id}"]
    kramdown = client.get_block_kramdown(block_id)
    parts.append("--- kramdown ---")
    parts.append(kramdown or "(empty)")
    if include_attrs:
        attrs = client.get_block_attrs(block_id)
        parts.append("--- attrs ---")
        if attrs:
            for key in sorted(attrs):
                parts.append(f"{key}={attrs[key]}")
        else:
            parts.append("(none)")
    blocks_seen = 1
    if include_children:
        children = client.get_child_blocks(block_id)
        parts.append(f"--- children ({len(children)}) ---")
        for child in children:
            if blocks_seen >= max_blocks:
                parts.append("(truncated)")
                break
            cid = str(child.get("id") or "")
            md = str(child.get("markdown") or child.get("content") or "")
            parts.append(f"- {cid}: {md[:500]}")
            blocks_seen += 1
    if resolve_refs and max_ref_depth > 0:
        refs = extract_block_refs(kramdown)
        parts.append(f"--- refs ({len(refs)}) depth={max_ref_depth} ---")
        for rid in refs:
            if blocks_seen >= max_blocks:
                parts.append("(truncated)")
                break
            ref_md = client.get_block_kramdown(rid)
            parts.append(f"### (({rid}))")
            parts.append((ref_md or "(empty)")[:3000])
            blocks_seen += 1
    return "\n".join(parts)


def siyuan_append_impl(
    cfg: AgentConfig,
    *,
    parent_id: str,
    content: str,
    as_result: bool = False,
    request_id: Optional[str] = None,
) -> str:
    """Append Markdown under *parent_id*. Returns deep link + new block id."""
    if not parent_id:
        return "Error: --parent-id is required."
    body = (content or "").strip()
    if not body:
        return "Error: content is empty."
    if as_result:
        rid = request_id or "unknown"
        body = build_result_markdown(body, request_id=rid)
    client = client_from_agent_config(cfg)
    new_id = client.append_block(parent_id, body)
    if not new_id:
        return "Error: SiYuan append returned no block id."
    url = build_siyuan_block_url(new_id)
    return f"Appended block_id={new_id} url={url}"


def siyuan_set_status_impl(
    cfg: AgentConfig,
    *,
    block_id: str,
    status: Optional[str] = None,
    request_id: Optional[str] = None,
    output_id: Optional[str] = None,
    source_hash: Optional[str] = None,
    run_id: Optional[str] = None,
) -> str:
    """Set allowed ``custom-agent-*`` attrs on *block_id*."""
    if not block_id:
        return "Error: --block-id is required."
    attrs: dict[str, str] = {}
    if status is not None:
        status_l = status.strip().lower()
        if status_l not in _ALLOWED_STATUS:
            return (
                f"Error: invalid status {status!r}. "
                f"Allowed: {', '.join(sorted(_ALLOWED_STATUS))}"
            )
        attrs[ATTR_STATUS] = status_l
    if request_id:
        attrs[ATTR_REQUEST_ID] = request_id
    if output_id:
        attrs[ATTR_OUTPUT_ID] = output_id
    if source_hash:
        attrs[ATTR_SOURCE_HASH] = source_hash
    if run_id:
        attrs[ATTR_RUN_ID] = run_id
    if not attrs:
        return "Error: provide at least one attribute to set."
    unknown = set(attrs) - _ALLOWED_ATTRS
    if unknown:
        return f"Error: disallowed attrs: {sorted(unknown)}"
    client = client_from_agent_config(cfg)
    client.set_block_attrs(block_id, attrs)
    rendered = " ".join(f"{k}={v}" for k, v in sorted(attrs.items()))
    return f"Updated attrs on {block_id}: {rendered}"
