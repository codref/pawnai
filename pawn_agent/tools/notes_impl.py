"""Vault note read/search/write and task note updates."""

from __future__ import annotations

import re

from pawn_agent.utils.config import AgentConfig
from pawn_core.vault import (
    VaultNotFound,
    VaultWriteDenied,
    dump_frontmatter,
    normalize_vault_key,
    parse_frontmatter,
    resolve_path_template,
)
from pawn_core.vault_config import vault_store_from_config

_WIKI_LINK_RE = re.compile(r"\[\[([^\]|#]+?)(?:#[^\]|]+)?(?:\|[^\]]+)?\]\]")
_RESULT_SECTION_RE = re.compile(
    r"(?is)(^##\s*Result\s*\n)(.*?)(?=^##\s|\Z)",
    re.MULTILINE,
)


def _agent_root(cfg: AgentConfig) -> str:
    return (getattr(cfg.vault, "agent_root", None) or "Pawn").rstrip("/")


def _resolve_wiki_candidates(from_key: str, link: str, agent_root: str) -> list[str]:
    name = (link or "").strip()
    if not name:
        return []
    if "/" in name:
        key = normalize_vault_key(name)
        return [key if key.endswith(".md") else f"{key}.md"]
    base_dir = from_key.rsplit("/", 1)[0] if "/" in from_key else ""
    root = agent_root.rstrip("/")
    candidates: list[str] = []
    if base_dir:
        candidates.append(normalize_vault_key(f"{base_dir}/{name}.md"))
    candidates.append(normalize_vault_key(f"{name}.md"))
    if root:
        candidates.append(normalize_vault_key(f"{root}/{name}.md"))
    seen: set[str] = set()
    out: list[str] = []
    for c in candidates:
        if c and c not in seen:
            seen.add(c)
            out.append(c)
    return out


def _read_note_body(store, key: str) -> tuple[str, str]:
    text = store.read(key)
    return key, text


def note_read_impl(cfg: AgentConfig, path: str, *, follow_links: int = 0) -> str:
    """Read *path* and optionally follow ``[[wiki links]]`` up to *follow_links* depth."""
    store = vault_store_from_config(cfg)
    key = normalize_vault_key(path)
    if not key:
        return "Error: empty vault path"
    try:
        _, text = _read_note_body(store, key)
    except VaultNotFound:
        return f"Error: note not found: {key!r}"
    except Exception as exc:
        return f"Error reading {key!r}: {exc}"

    parts: list[str] = [f"# Vault: {key}\n", text.rstrip()]
    if follow_links <= 0:
        return "\n".join(parts).rstrip() + "\n"

    agent_root = _agent_root(cfg)
    visited: set[str] = {key}
    frontier: list[tuple[str, int]] = [(key, 0)]
    while frontier:
        current_key, depth = frontier.pop(0)
        if depth >= follow_links:
            continue
        try:
            body = store.read(current_key)
        except VaultNotFound:
            continue
        except Exception:
            continue
        for match in _WIKI_LINK_RE.finditer(body):
            link_name = match.group(1).strip()
            for target in _resolve_wiki_candidates(current_key, link_name, agent_root):
                if target in visited:
                    continue
                visited.add(target)
                try:
                    _, linked = _read_note_body(store, target)
                except VaultNotFound:
                    continue
                except Exception:
                    continue
                parts.append(f"\n---\n\n# Vault: {target}\n\n{linked.rstrip()}")
                frontier.append((target, depth + 1))
                break

    return "\n".join(parts).rstrip() + "\n"


def _note_matches_tag(text: str, tag: str) -> bool:
    needle = tag.strip().lstrip("#").lower()
    if not needle:
        return True
    meta, body = parse_frontmatter(text)
    tags = meta.get("tags")
    if isinstance(tags, list):
        for t in tags:
            if str(t).strip().lstrip("#").lower() == needle:
                return True
    elif isinstance(tags, str) and tags.strip().lstrip("#").lower() == needle:
        return True
    pattern = rf"(?<![\w/])#{re.escape(needle)}\b"
    return re.search(pattern, body, re.IGNORECASE) is not None


def note_search_impl(
    cfg: AgentConfig,
    *,
    folder: str = "",
    tag: str = "",
    limit: int = 50,
) -> str:
    """List vault keys under *folder*, optionally filtered by frontmatter or inline tag."""
    store = vault_store_from_config(cfg)
    agent_root = _agent_root(cfg)
    prefix = normalize_vault_key(folder) if folder else agent_root
    try:
        keys = store.list(prefix)
    except Exception as exc:
        return f"Error listing {prefix!r}: {exc}"

    cap = max(1, int(limit))
    if not tag.strip():
        shown = keys[:cap]
        if not shown:
            return f"(no notes under {prefix!r})"
        return "\n".join(f"- {k}" for k in shown) + "\n"

    matched: list[str] = []
    for key in keys:
        if len(matched) >= cap:
            break
        try:
            text = store.read(key)
        except Exception:
            continue
        if _note_matches_tag(text, tag):
            matched.append(key)

    if not matched:
        return f"(no notes under {prefix!r} matching tag {tag!r})"
    return "\n".join(f"- {k}" for k in matched) + "\n"


def note_write_impl(
    cfg: AgentConfig,
    path: str,
    content: str,
    *,
    create_only: bool = False,
) -> str:
    """Write *content* to *path* via :class:`VaultStore`."""
    store = vault_store_from_config(cfg)
    key = normalize_vault_key(path)
    if not key:
        return "Error: empty vault path"
    try:
        if create_only and store.exists(key):
            return f"Error: note already exists: {key!r} (create_only)"
        stat = store.write(key, content)
        return f"wrote {key} (etag={stat.etag})"
    except VaultWriteDenied as exc:
        return f"Error: write denied: {exc}"
    except Exception as exc:
        return f"Error writing {key!r}: {exc}"


def note_append_impl(cfg: AgentConfig, path: str, content: str) -> str:
    """Append *content* to an existing note (or create it)."""
    store = vault_store_from_config(cfg)
    key = normalize_vault_key(path)
    if not key:
        return "Error: empty vault path"
    try:
        stat = store.append(key, content)
        return f"appended {key} (etag={stat.etag})"
    except VaultWriteDenied as exc:
        return f"Error: append denied: {exc}"
    except Exception as exc:
        return f"Error appending {key!r}: {exc}"


def _task_note_key(cfg: AgentConfig, task_id: str) -> str:
    raw = (task_id or "").strip()
    if not raw:
        raise ValueError("empty task_id")
    if raw.endswith(".md") or "/" in raw:
        return normalize_vault_key(raw)
    vault = cfg.vault
    return resolve_path_template(
        vault.task_path_template,
        agent_root=vault.agent_root,
        task_id=raw,
    )


def replace_result_section(body: str, result: str) -> str:
    """Replace or append the ``## Result`` section in *body*."""
    text = body if body.endswith("\n") or not body else body + "\n"
    result_body = result.rstrip() + "\n"
    match = _RESULT_SECTION_RE.search(text)
    if match:
        start, end = match.span(2)
        return text[:start] + result_body + text[end:]
    if not text.strip():
        return f"## Result\n\n{result_body}"
    return text.rstrip() + f"\n\n## Result\n\n{result_body}"


def task_update_impl(
    cfg: AgentConfig,
    task_id: str,
    *,
    status: str | None = None,
    result: str | None = None,
) -> str:
    """Update frontmatter *status* and/or ``## Result`` on a task note."""
    if status is None and result is None:
        return "Error: provide --status and/or --result"
    store = vault_store_from_config(cfg)
    try:
        key = _task_note_key(cfg, task_id)
    except ValueError as exc:
        return f"Error: {exc}"

    try:
        existing = store.read(key)
    except VaultNotFound:
        return f"Error: task note not found: {key!r}"
    except Exception as exc:
        return f"Error reading {key!r}: {exc}"

    meta, body = parse_frontmatter(existing)
    if status is not None:
        meta["status"] = status.strip()
    if result is not None:
        body = replace_result_section(body, result)

    updated = dump_frontmatter(meta, body)
    try:
        stat = store.write(key, updated)
    except VaultWriteDenied as exc:
        return f"Error: write denied: {exc}"
    except Exception as exc:
        return f"Error writing {key!r}: {exc}"

    parts = [f"updated {key} (etag={stat.etag})"]
    if status is not None:
        parts.append(f"status={status!r}")
    if result is not None:
        parts.append("result section replaced")
    return "; ".join(parts)
