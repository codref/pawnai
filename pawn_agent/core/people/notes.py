"""Person note read/write with managed vs preserved sections.

Contract (keep stable — Obsidian users rely on these headings):

- Path: ``{people_dir}/{speaker_id}.md`` (default ``People/davide.md``).
- Frontmatter: ``pawn: person``, ``speaker_id``, ``aliases``, ``tags``, …
- Managed: title, ``## Summary``, ``## Facts``, ``## Appearances``,
  optional ``## Open loops``.
- Preserved: ``## Notes`` — never overwritten by Pawn.

Writes use ``skip_guards=True`` because notes live outside ``Pawn/``.
Free ``note_write`` must not edit these files (no ``pawn: editable``).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Any, Optional, Sequence

from pawn_agent.utils.config import AgentConfig
from pawn_core.vault import VaultNotFound, dump_frontmatter, normalize_vault_key, parse_frontmatter
from pawn_core.vault_config import vault_store_from_config

# Section splitter: anything under ## Notes is preserved across updates.
_SECTION_RE = re.compile(r"(?m)^(## .+)$")
_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')


@dataclass
class PersonNote:
    """Parsed person note ready for tool output or further edits."""

    key: str
    speaker_id: str
    display_name: str
    aliases: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    status: str = "active"
    me: bool = False
    summary: str = ""
    facts: list[str] = field(default_factory=list)
    appearances: list[str] = field(default_factory=list)
    open_loops: list[str] = field(default_factory=list)
    notes: str = ""  # user-owned body under ## Notes
    meta: dict[str, Any] = field(default_factory=dict)
    raw: str = ""


def people_dir(cfg: AgentConfig) -> str:
    """Return the configured people folder without a trailing slash."""
    raw = (getattr(cfg.coworker, "people_dir", None) or "People").strip().strip("/")
    return raw or "People"


def people_note_key(cfg: AgentConfig, speaker_id: str) -> str:
    """Stable vault key ``People/{speaker_id}.md``."""
    sid = _safe_id(speaker_id)
    return normalize_vault_key(f"{people_dir(cfg)}/{sid}.md")


def people_wiki_target(cfg: AgentConfig, speaker_id: str) -> str:
    """Wikilink target without ``[[…]]`` (Obsidian prefers path without ``.md``)."""
    sid = _safe_id(speaker_id)
    return f"{people_dir(cfg)}/{sid}"


def _safe_id(speaker_id: str) -> str:
    cleaned = _UNSAFE.sub("-", (speaker_id or "").strip()).strip("-")
    return cleaned or "unknown"


def _today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def _split_sections(body: str) -> tuple[str, dict[str, str]]:
    """Split markdown body into (preamble, {heading: content}).

    *heading* is the text after ``## `` (e.g. ``Summary``).
    """
    text = body or ""
    parts = _SECTION_RE.split(text)
    preamble = parts[0].strip() if parts else ""
    sections: dict[str, str] = {}
    i = 1
    while i + 1 < len(parts):
        heading_line = parts[i].strip()
        content = parts[i + 1]
        name = heading_line[3:].strip() if heading_line.startswith("## ") else heading_line
        sections[name] = content.strip("\n")
        i += 2
    return preamble, sections


def _bullet_lines(section_body: str) -> list[str]:
    """Return non-empty bullet texts (leading ``- `` stripped)."""
    out: list[str] = []
    for line in (section_body or "").splitlines():
        stripped = line.strip()
        if stripped.startswith("- "):
            out.append(stripped[2:].strip())
        elif stripped.startswith("* "):
            out.append(stripped[2:].strip())
    return out


def _bullets_block(items: Sequence[str]) -> str:
    lines = [f"- {item.strip()}" for item in items if (item or "").strip()]
    return "\n".join(lines)


def render_person_note(
    *,
    speaker_id: str,
    display_name: str,
    aliases: Optional[Sequence[str]] = None,
    tags: Optional[Sequence[str]] = None,
    status: str = "active",
    me: bool = False,
    summary: str = "",
    facts: Optional[Sequence[str]] = None,
    appearances: Optional[Sequence[str]] = None,
    open_loops: Optional[Sequence[str]] = None,
    notes: str = "",
    updated: Optional[str] = None,
) -> str:
    """Build a full person note from structured fields."""
    alias_list = [a.strip() for a in (aliases or []) if str(a).strip()]
    tag_list = _normalize_tags(tags)
    meta: dict[str, Any] = {
        "pawn": "person",
        "speaker_id": speaker_id,
        "aliases": alias_list,
        "tags": tag_list,
        "status": status or "active",
        "updated": updated or _today_iso(),
    }
    if me:
        meta["me"] = True

    parts = [
        f"# {display_name.strip() or speaker_id}",
        "",
        "## Summary",
        (summary or "").strip() or "_No summary yet._",
        "",
        "## Facts",
        _bullets_block(facts or []) or "_No facts yet._",
        "",
        "## Appearances",
        _bullets_block(appearances or []) or "_No appearances yet._",
        "",
    ]
    loops = list(open_loops or [])
    if loops:
        parts.extend(["## Open loops", _bullets_block(loops), ""])
    # Always emit Notes so the user has a stable place to write.
    notes_body = (
        notes or ""
    ).strip() or "_Add private notes here. Pawn will not overwrite this section._"
    parts.extend(["## Notes", notes_body, ""])
    return dump_frontmatter(meta, "\n".join(parts))


def parse_person_note(text: str, *, key: str = "") -> PersonNote:
    """Parse a person note. Raises ValueError when ``pawn`` is not ``person``."""
    meta, body = parse_frontmatter(text or "")
    if meta.get("pawn") != "person":
        raise ValueError("frontmatter pawn must be 'person'")
    speaker_id = str(meta.get("speaker_id") or "").strip()
    preamble, sections = _split_sections(body)
    # Title from first H1 in preamble, else speaker_id.
    display = speaker_id
    for line in preamble.splitlines():
        if line.startswith("# "):
            display = line[2:].strip() or display
            break
    aliases = meta.get("aliases") or []
    if not isinstance(aliases, list):
        aliases = [str(aliases)]
    tags = meta.get("tags") or []
    if not isinstance(tags, list):
        tags = [str(tags)]
    summary = sections.get("Summary", "").strip()
    if summary == "_No summary yet._":
        summary = ""
    notes = sections.get("Notes", "")
    if notes.strip() == "_Add private notes here. Pawn will not overwrite this section._":
        notes = ""
    return PersonNote(
        key=key,
        speaker_id=speaker_id,
        display_name=display,
        aliases=[str(a).strip() for a in aliases if str(a).strip()],
        tags=_normalize_tags(tags),
        status=str(meta.get("status") or "active").strip().lower(),
        me=bool(meta.get("me")),
        summary=summary,
        facts=_bullet_lines(sections.get("Facts", "")),
        appearances=_bullet_lines(sections.get("Appearances", "")),
        open_loops=_bullet_lines(sections.get("Open loops", "")),
        notes=notes.strip("\n"),
        meta=dict(meta),
        raw=text or "",
    )


def _normalize_tags(tags: Optional[Sequence[Any]]) -> list[str]:
    """Ensure ``person`` is present; strip leading ``#``; de-dupe case-insensitively."""
    seen: set[str] = set()
    out: list[str] = []
    for raw in list(tags or []) + ["person"]:
        tag = str(raw or "").strip().lstrip("#")
        if not tag:
            continue
        key = tag.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(tag)
    # Keep ``person`` first for Obsidian filter convenience.
    if "person" in {t.lower() for t in out}:
        out = ["person"] + [t for t in out if t.lower() != "person"]
    return out


def ensure_person_note(
    cfg: AgentConfig,
    *,
    speaker_id: str,
    display_name: str,
    aliases: Optional[Sequence[str]] = None,
    me: bool = False,
    store: Any = None,
) -> tuple[str, bool]:
    """Create a stub person note when missing.

    Returns ``(vault_key, created)``. Existing notes are left untouched.
    """
    vault = store if store is not None else vault_store_from_config(cfg)
    key = people_note_key(cfg, speaker_id)
    try:
        vault.read(key)
    except VaultNotFound:
        body = render_person_note(
            speaker_id=_safe_id(speaker_id),
            display_name=display_name or speaker_id,
            aliases=aliases,
            me=me,
        )
        vault.write(key, body, skip_guards=True)
        return key, True
    return key, False


def read_person_note(
    cfg: AgentConfig,
    speaker_id: str,
    *,
    store: Any = None,
) -> Optional[PersonNote]:
    """Load and parse a person note, or ``None`` when missing."""
    vault = store if store is not None else vault_store_from_config(cfg)
    key = people_note_key(cfg, speaker_id)
    try:
        text = vault.read(key)
    except VaultNotFound:
        return None
    try:
        return parse_person_note(text, key=key)
    except ValueError:
        return None


def update_person_note(
    cfg: AgentConfig,
    *,
    speaker_id: str,
    display_name: Optional[str] = None,
    summary: Optional[str] = None,
    aliases: Optional[Sequence[str]] = None,
    tags: Optional[Sequence[str]] = None,
    add_facts: Optional[Sequence[str]] = None,
    add_appearances: Optional[Sequence[str]] = None,
    add_open_loops: Optional[Sequence[str]] = None,
    status: Optional[str] = None,
    me: Optional[bool] = None,
    store: Any = None,
) -> str:
    """Merge managed fields into an existing note (or create one).

    ``## Notes`` from the existing file is always preserved.
    Facts / appearances / open loops are append-only with de-dupe.
    """
    vault = store if store is not None else vault_store_from_config(cfg)
    key = people_note_key(cfg, speaker_id)
    existing: Optional[PersonNote] = None
    try:
        existing = parse_person_note(vault.read(key), key=key)
    except VaultNotFound:
        existing = None
    except ValueError as exc:
        raise ValueError(f"{key} exists but is not a pawn: person note") from exc

    if existing is None:
        ensure_person_note(
            cfg,
            speaker_id=speaker_id,
            display_name=display_name or speaker_id,
            aliases=aliases,
            me=bool(me),
            store=vault,
        )
        existing = read_person_note(cfg, speaker_id, store=vault)
        if existing is None:
            raise RuntimeError(f"failed to create person note {key}")

    name = (display_name or existing.display_name or speaker_id).strip()
    new_aliases = list(existing.aliases)
    if aliases is not None:
        for a in aliases:
            a = str(a).strip()
            if a and a.lower() not in {x.lower() for x in new_aliases}:
                new_aliases.append(a)
    new_tags = _normalize_tags(list(existing.tags) + list(tags or []))
    facts = _dedupe_append(existing.facts, add_facts or [])
    appearances = _dedupe_append(existing.appearances, add_appearances or [])
    loops = _dedupe_append(existing.open_loops, add_open_loops or [])
    new_summary = existing.summary
    if summary is not None and summary.strip():
        new_summary = summary.strip()

    body = render_person_note(
        speaker_id=existing.speaker_id or _safe_id(speaker_id),
        display_name=name,
        aliases=new_aliases,
        tags=new_tags,
        status=status or existing.status,
        me=existing.me if me is None else bool(me),
        summary=new_summary,
        facts=facts,
        appearances=appearances,
        open_loops=loops,
        notes=existing.notes,
        updated=_today_iso(),
    )
    vault.write(key, body, skip_guards=True)
    return key


def append_facts(
    cfg: AgentConfig,
    speaker_id: str,
    facts: Sequence[str],
    *,
    source_wiki: str = "",
    store: Any = None,
) -> str:
    """Append dated fact bullets, optionally suffixing a source wikilink."""
    day = date.today().isoformat()
    prepared: list[str] = []
    for fact in facts:
        text = (fact or "").strip()
        if not text:
            continue
        if source_wiki and source_wiki not in text:
            text = f"{day}: {text} ({source_wiki})"
        elif not text[:10].count("-") == 2:
            text = f"{day}: {text}"
        prepared.append(text)
    return update_person_note(cfg, speaker_id=speaker_id, add_facts=prepared, store=store)


def _dedupe_append(existing: Sequence[str], additions: Sequence[str]) -> list[str]:
    seen = {e.strip().lower() for e in existing if e.strip()}
    out = [e.strip() for e in existing if e.strip()]
    for item in additions:
        text = (item or "").strip()
        if not text:
            continue
        key = text.lower()
        if key in seen:
            continue
        # Also skip when the new line is a pure substring of an existing bullet
        # (common when re-running refresh with the same appearance wiki).
        if any(key in s or s in key for s in seen):
            continue
        seen.add(key)
        out.append(text)
    return out


def format_person_card(note: PersonNote) -> str:
    """Human-readable tool observation for one person."""
    lines = [
        f"{note.display_name}  id={note.speaker_id}  status={note.status}",
        f"key: {note.key}",
    ]
    if note.aliases:
        lines.append(f"aliases: {', '.join(note.aliases)}")
    if note.tags:
        lines.append(f"tags: {', '.join(note.tags)}")
    if note.me:
        lines.append("me: true")
    if note.summary:
        lines.append("")
        lines.append("## Summary")
        lines.append(note.summary)
    if note.facts:
        lines.append("")
        lines.append("## Facts")
        for f in note.facts[-12:]:
            lines.append(f"- {f}")
    if note.appearances:
        lines.append("")
        lines.append("## Appearances")
        for a in note.appearances[-12:]:
            lines.append(f"- {a}")
    if note.notes.strip():
        lines.append("")
        lines.append("## Notes (user)")
        lines.append(note.notes.strip())
    return "\n".join(lines)
