"""Obsidian vault task note protocol (pure helpers)."""

from __future__ import annotations

import hashlib
import re
import urllib.parse
from typing import Any, Mapping, Sequence

from pawn_core.vault import dump_frontmatter, normalize_vault_key, parse_frontmatter

_SECTION_INSTRUCTION = "Instruction"
_SECTION_CONTEXT = "Context"
_SECTION_RESULT = "Result"
_SECTION_NAMES = (_SECTION_INSTRUCTION, _SECTION_CONTEXT, _SECTION_RESULT)

_WIKI_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")
_SECTION_RE = re.compile(
    r"(?m)^##\s+(Instruction|Context|Result)\s*\n(.*?)(?=^##\s+|\Z)",
    re.DOTALL,
)

_TASK_STATUSES = frozenset({"todo", "running", "review", "done", "blocked"})


def instruction_hash(text: str) -> str:
    """Stable hash of normalized instruction text."""
    normalized = re.sub(r"\s+", " ", (text or "").strip()).lower()
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:32]


def conversation_id_for_note(path: str) -> str:
    """Sallm conversation key for a vault note path."""
    return f"note:{normalize_vault_key(path)}"


def extract_wiki_links(text: str) -> list[str]:
    """Return unique wiki link targets from Obsidian ``[[link]]`` syntax."""
    seen: set[str] = set()
    out: list[str] = []
    for match in _WIKI_LINK_RE.finditer(text or ""):
        target = normalize_vault_key(match.group(1).strip())
        if target and target not in seen:
            seen.add(target)
            out.append(target)
    return out


def wiki_link_to_vault_key(link: str) -> str:
    """Map a wiki link target to a vault key (add ``.md`` when missing)."""
    key = normalize_vault_key(link)
    if not key:
        return key
    if not key.endswith(".md"):
        key = f"{key}.md"
    return key


def build_obsidian_open_url(vault_name: str, file_path: str) -> str:
    """Build an ``obsidian://open`` URL for *file_path* in *vault_name*."""
    vault = urllib.parse.quote(vault_name or "", safe="")
    file_norm = normalize_vault_key(file_path).replace("/", "%2F")
    return f"obsidian://open?vault={vault}&file={file_norm}"


def get_section(body: str, name: str) -> str:
    """Return the text under ``## {name}`` or empty string."""
    if name not in _SECTION_NAMES:
        return ""
    for match in _SECTION_RE.finditer(body or ""):
        if match.group(1) == name:
            return match.group(2).strip("\n")
    return ""


def set_section(body: str, name: str, content: str) -> str:
    """Replace or append a ``## {name}`` section in *body*."""
    if name not in _SECTION_NAMES:
        return body
    text = body or ""
    if not text.endswith("\n") and text:
        text += "\n"
    new_block = f"## {name}\n{(content or '').strip()}\n"
    pattern = re.compile(
        rf"(?m)^##\s+{re.escape(name)}\s*\n.*?(?=^##\s+|\Z)",
        re.DOTALL,
    )
    if pattern.search(text):
        return pattern.sub(new_block, text, count=1)
    if not text.strip():
        return new_block
    return text.rstrip() + "\n\n" + new_block


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def parse_task_note(text: str) -> dict[str, Any]:
    """Parse a ``pawn: task`` note into structured fields."""
    meta, body = parse_frontmatter(text or "")
    if meta.get("pawn") != "task":
        raise ValueError("frontmatter pawn must be 'task'")
    task_id = str(meta.get("id") or "").strip()
    status = str(meta.get("status") or "todo").strip().lower()
    if status not in _TASK_STATUSES:
        status = "todo"
    conversation = str(meta.get("conversation") or "").strip()
    note_path = meta.get("note")
    if isinstance(note_path, str):
        note_path = note_path.strip().strip('"').strip("'")
        if note_path.startswith("[[") and note_path.endswith("]]"):
            note_path = note_path[2:-2].split("|", 1)[0].strip()
    else:
        note_path = None
    return {
        "id": task_id,
        "status": status,
        "approved": _coerce_bool(meta.get("approved")),
        "conversation": conversation,
        "note_path": note_path,
        "instruction": get_section(body, _SECTION_INSTRUCTION),
        "context": get_section(body, _SECTION_CONTEXT),
        "result": get_section(body, _SECTION_RESULT),
        "meta": meta,
        "body": body,
    }


def render_task_note(
    *,
    task_id: str,
    status: str,
    instruction: str,
    context: str = "",
    result: str = "",
    conversation: str = "",
    note_path: str | None = None,
    approved: bool = False,
    extra_meta: Mapping[str, Any] | None = None,
) -> str:
    """Serialize a task note with YAML frontmatter and standard sections."""
    meta: dict[str, Any] = {
        "pawn": "task",
        "id": task_id,
        "status": status,
        "approved": approved,
    }
    if conversation:
        meta["conversation"] = conversation
    if note_path:
        meta["note"] = f'"[[{note_path}]]"' if " " in note_path else f"[[{note_path}]]"
    if extra_meta:
        for key, value in extra_meta.items():
            if key not in meta:
                meta[key] = value
    body = ""
    body = set_section(body, _SECTION_INSTRUCTION, instruction)
    if context.strip():
        body = set_section(body, _SECTION_CONTEXT, context)
    if result.strip():
        body = set_section(body, _SECTION_RESULT, result)
    return dump_frontmatter(meta, body)


def build_agent_prompt(
    *,
    task_id: str,
    instruction: str,
    task_context: str = "",
    note_excerpt: str = "",
    linked_excerpts: Sequence[tuple[str, str]] = (),
    task_key: str = "",
    note_path: str | None = None,
) -> str:
    """Compose the prompt for a ``vault_run`` agent turn."""
    lines = [
        "Obsidian vault task — fulfill the instruction below.",
        f"task_id: {task_id}",
    ]
    if task_key:
        lines.append(f"task_key: {task_key}")
    if note_path:
        lines.append(f"linked_note: {note_path}")
    lines.extend(
        [
            "",
            "Rules:",
            "- Use vault tools / sessions tools as appropriate for diarization ids.",
            "- Do not invent session ids; discover via sessions_list.",
            "- Return a clear markdown result for the ## Result section.",
            "- Do not write tool errors into the vault.",
            "",
            "## Instruction",
            (instruction or "").strip() or "(empty)",
        ]
    )
    if task_context.strip():
        lines.extend(["", "## Task context", task_context.strip()[:8000]])
    if note_excerpt.strip():
        lines.extend(["", "## Linked note excerpt", note_excerpt.strip()[:8000]])
    if linked_excerpts:
        lines.append("")
        lines.append("## Wiki-linked notes")
        for link, excerpt in linked_excerpts:
            lines.append(f"### [[{link}]]")
            lines.append((excerpt or "")[:4000])
    return "\n".join(lines)
