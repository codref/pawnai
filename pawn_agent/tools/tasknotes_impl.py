"""TaskNotes notes, pick-lists, and boards.

Tasks are ordinary Markdown under the agent root, tagged so the TaskNotes
plugin shows them on its kanban and calendar. Pawn does not call the plugin
HTTP API and does not hold calendar OAuth tokens.

Usability this module is built around:

- A proposal note is a checklist the user can edit. Checked lines are created.
  ``--pick`` is an explicit override. ``--all`` includes unchecked lines.
- The same title and assignee are one task, even across sessions. A repeat
  extraction skips the existing note instead of cloning it.
- ``scheduled`` is a local wall time with no timezone suffix, so the calendar
  does not shift it. Empty dates stay empty.
- ``me`` and configured aliases collapse to one display name.
- Boards are ``.base`` files. A file the user has customized (marker removed)
  is not overwritten.
- Notes in ``external_tasks_dir`` (the plugin's own folder) are listed and
  deduped, and are not rewritten.
"""

from __future__ import annotations

import hashlib
import json
import re
import uuid
from datetime import datetime, timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from pawn_agent.utils.config import AgentConfig
from pawn_core.vault import (
    VaultError,
    dump_frontmatter,
    is_under_agent_root,
    normalize_vault_key,
    parse_frontmatter,
)
from pawn_core.vault_config import vault_store_from_config

_BOARD_MARKER = "# pawn-tasknotes-board"
_FENCE_RE = re.compile(r"```tasknotes\s*\n(.*?)\n```", re.DOTALL)
_CHECK_RE = re.compile(r"^- \[(?P<mark>[ xX])\] `(?P<id>[^`]+)` (?P<rest>.*)$")
_WIKI_ONLY_RE = re.compile(r"^\[\[[^\]]+\]\]$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_TIME_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}(?::\d{2})?$")
_TAG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_/-]{0,40}$")
_LABELS = ("assignee", "due", "scheduled", "project", "priority", "status", "estimate")
_PRIORITIES = {
    "none": "none",
    "low": "low",
    "normal": "normal",
    "medium": "normal",
    "high": "high",
    "urgent": "high",
}
_STATUSES = {
    "open": "open",
    "todo": "open",
    "to-do": "open",
    "none": "open",
    "in-progress": "in-progress",
    "in progress": "in-progress",
    "doing": "in-progress",
    "wip": "in-progress",
    "done": "done",
    "complete": "done",
    "completed": "done",
}
_BOARDS = {"", "none", "assignee", "project", "both"}
_UNSAFE_STEM = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_ME = {"me", "i", "myself", "mine"}
_PERSON_BOARD_RE = re.compile(r"^(?P<who>.+?)['’]s\s+boards?$", re.IGNORECASE)

_PENDING_INTRO = """\
Nothing has been added to TaskNotes yet.

- Leave a box checked to create that task. Uncheck it to skip it.
- Edit the title and any labeled field on the line (assignee, due, scheduled, project, priority, status, estimate).
- Due is a date (`2026-10-02`). Scheduled is that date, or a local time with no timezone (`2026-10-02T09:00`). Leave both off when nobody named a day.
- The indented text becomes the task note. A wikilink alone on an indented line is kept as a link, not as the note body.
- Ask Pawn to create the checked tasks, or reply with the numbers you want.

Times below are in {tz}. A task with the same title and assignee that is already on the board is skipped, not duplicated.
"""


class _Dumper(yaml.SafeDumper):
    """Emit JSON-style booleans so Bases files stay portable."""


def _repr_bool(dumper: yaml.SafeDumper, data: bool) -> Any:
    return dumper.represent_scalar("tag:yaml.org,2002:bool", "true" if data else "false")


_Dumper.add_representer(bool, _repr_bool)


def _now(cfg: AgentConfig, now: datetime | None) -> datetime:
    current = now or datetime.now(_tz(cfg))
    if current.tzinfo is None:
        current = current.replace(tzinfo=_tz(cfg))
    return current


def _tz(cfg: AgentConfig) -> ZoneInfo:
    name = (cfg.tasknotes.timezone or cfg.coworker.timezone or "UTC").strip() or "UTC"
    try:
        return ZoneInfo(name)
    except ZoneInfoNotFoundError:
        return ZoneInfo("UTC")


def _tz_name(cfg: AgentConfig) -> str:
    return getattr(_tz(cfg), "key", None) or "UTC"


def _root(cfg: AgentConfig) -> str:
    return normalize_vault_key(cfg.vault.agent_root or "Pawn").rstrip("/") or "Pawn"


def _safe_key(path: str) -> str:
    key = normalize_vault_key(path)
    parts = [p for p in key.split("/") if p != ""]
    if not parts or any(p in {".", ".."} for p in parts):
        raise ValueError(f"invalid vault path: {path!r}")
    return "/".join(parts)


def _under_root(cfg: AgentConfig, key: str, *, what: str) -> str:
    root = _root(cfg)
    try:
        cleaned = _safe_key(key)
    except ValueError as exc:
        raise ValueError(f"{what}: {exc}") from exc
    if not is_under_agent_root(cleaned, root):
        raise ValueError(
            f"{what} {cleaned!r} is outside {root}/. "
            "TaskNotes files Pawn writes have to stay in the agent root."
        )
    return cleaned


def _leaf_dir(cfg: AgentConfig, configured: str, leaf: str) -> str:
    raw = (configured or "").strip()
    if raw:
        return _under_root(cfg, raw, what=f"tasknotes {leaf} directory")
    return _under_root(cfg, f"{_root(cfg)}/TaskNotes/{leaf}", what=f"tasknotes {leaf} directory")


def _dirs(cfg: AgentConfig) -> dict[str, str]:
    tn = cfg.tasknotes
    return {
        "tasks": _leaf_dir(cfg, tn.tasks_dir, "Tasks"),
        "views": _leaf_dir(cfg, tn.views_dir, "Views"),
        "projects": _leaf_dir(cfg, tn.projects_dir, "Projects"),
        "proposals": _leaf_dir(cfg, tn.proposals_dir, "Proposals"),
    }


def _ident_tag(cfg: AgentConfig) -> str:
    tag = (cfg.tasknotes.ident_tag or "task").strip().lstrip("#")
    if not _TAG_RE.fullmatch(tag):
        raise ValueError(f"tasknotes.ident_tag {tag!r} is not a single tag token")
    return tag


def _display_name(cfg: AgentConfig) -> str:
    explicit = (cfg.tasknotes.display_name or "").strip()
    if explicit:
        return explicit
    for name in _me_names(cfg):
        if name.strip():
            return name.strip()
    return "Me"


def _me_names(cfg: AgentConfig) -> list[str]:
    names = [n.strip() for n in cfg.tasknotes.me if n and n.strip()]
    if names:
        return names
    return [n.strip() for n in cfg.coworker.me if n and n.strip()]


def person_from_board_name(name: str) -> str:
    """``Edo's board`` is the person Edo, not a note titled Edo's Board."""
    text = " ".join((name or "").replace("’", "'").split())
    match = _PERSON_BOARD_RE.match(text)
    if match is None:
        return ""
    return match.group("who").strip(" -")


def _canonicalize_assignee(raw: str, cfg: AgentConfig) -> str:
    text = " ".join((raw or "").split())
    if not text:
        return ""
    owner = person_from_board_name(text)
    if owner:
        text = owner
    folded = text.casefold()
    aliases = _ME | {n.casefold() for n in _me_names(cfg)}
    if folded in aliases:
        return _display_name(cfg)
    return text


def _safe_stem(title: str) -> str:
    cleaned = _UNSAFE_STEM.sub(" ", title)
    cleaned = re.sub(r"\s+", " ", cleaned).strip().rstrip(".")
    if len(cleaned) > 80:
        cleaned = cleaned[:80].rstrip()
    return cleaned or "task"


def _fingerprint(title: str, assignee: str) -> str:
    blob = f"{' '.join(title.casefold().split())}|{assignee.casefold().strip()}"
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def _stamp(moment: datetime) -> str:
    return moment.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _today(moment: datetime) -> str:
    return moment.date().isoformat()


def _check_due(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if not _DATE_RE.fullmatch(text):
        raise ValueError(f"due must be YYYY-MM-DD, got {text!r}")
    datetime.strptime(text, "%Y-%m-%d")
    return text


def _check_scheduled(value: str) -> str:
    text = (value or "").strip()
    if not text:
        return ""
    if text.endswith("Z") or re.search(r"[+-]\d{2}:\d{2}$", text):
        raise ValueError(
            "scheduled must be a local wall time with no timezone suffix "
            f"(YYYY-MM-DD or YYYY-MM-DDTHH:MM), got {text!r}"
        )
    if _DATE_RE.fullmatch(text):
        datetime.strptime(text, "%Y-%m-%d")
        return text
    if not _TIME_RE.fullmatch(text):
        raise ValueError(
            "scheduled must be YYYY-MM-DD or YYYY-MM-DDTHH:MM with no timezone, " f"got {text!r}"
        )
    datetime.strptime(text[:10], "%Y-%m-%d")
    hh, mm = text[11:16].split(":")
    if not (0 <= int(hh) <= 23 and 0 <= int(mm) <= 59):
        raise ValueError(f"scheduled clock time is invalid: {text!r}")
    if len(text) == 19:
        sec = int(text[17:19])
        if not 0 <= sec <= 59:
            raise ValueError(f"scheduled clock time is invalid: {text!r}")
    return text


def _check_status(value: str, *, default: str) -> str:
    text = (value or "").strip().casefold()
    if not text:
        text = default
    if text not in _STATUSES:
        allowed = "open, in-progress, done"
        raise ValueError(f"status must be {allowed}, got {value!r}")
    return _STATUSES[text]


def _check_priority(value: str, *, default: str) -> str:
    text = (value or "").strip().casefold()
    if not text:
        text = default
    if text not in _PRIORITIES:
        raise ValueError(f"priority must be none, low, normal, or high, got {value!r}")
    return _PRIORITIES[text]


def _check_estimate(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        minutes = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"time_estimate must be minutes, got {value!r}") from exc
    if minutes <= 0:
        return None
    if minutes > 10000:
        raise ValueError(f"time_estimate {minutes} is too large; pass minutes, not seconds")
    return minutes


def _as_list(value: Any) -> list[str]:
    if value is None or value == "":
        return []
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    if isinstance(value, (list, tuple)):
        return [str(part).strip() for part in value if str(part).strip()]
    raise ValueError(f"expected a list or string, got {value!r}")


def _clean_title(value: str) -> str:
    title = " ".join((value or "").replace("·", " ").replace("`", " ").split())
    if not title:
        raise ValueError("title is empty")
    if len(title) > 200:
        raise ValueError("title is longer than 200 characters")
    folded = title.casefold()
    if folded.startswith("error:") or "traceback (most recent call last)" in folded:
        raise ValueError("title looks like a tool error, not a task")
    return title


def _boards_mode(value: str) -> str:
    text = (value or "").strip().casefold()
    if text not in _BOARDS:
        raise ValueError("boards must be none, assignee, project, or both")
    return "" if text in {"", "none"} else text


def normalize_item(raw: Any, cfg: AgentConfig) -> dict[str, Any]:
    """Validate one task dict and fill status, priority, and assignee."""
    if not isinstance(raw, dict):
        raise ValueError("item must be a JSON object")
    title = _clean_title(str(raw.get("title") or ""))
    item_id = str(raw.get("id") or "").strip().replace("`", "")
    if not item_id:
        raise ValueError("item id is empty")
    contexts = _as_list(raw.get("contexts"))
    if len(contexts) > 8:
        raise ValueError("at most 8 contexts")
    for ctx in contexts:
        if len(ctx) > 40:
            raise ValueError(f"context {ctx!r} is too long")
    blocked = [str(part).strip() for part in _as_list(raw.get("blocked_by"))]
    details = str(raw.get("details") or "").replace("```", "'''").strip()
    if len(details) > 4000:
        raise ValueError("details are longer than 4000 characters")
    source = " ".join(str(raw.get("source") or "").split())
    source_note = str(raw.get("source_note") or "").strip()
    if source_note:
        source_note = _safe_key(source_note)
        if not source_note.endswith(".md"):
            source_note += ".md"
    project = " ".join(str(raw.get("project") or "").replace("\n", " ").split())
    if len(project) > 120:
        raise ValueError("project name is too long")
    selected = raw.get("selected", True)
    return {
        "id": item_id,
        "title": title,
        "details": details,
        "status": _check_status(str(raw.get("status") or ""), default=cfg.tasknotes.default_status),
        "priority": _check_priority(
            str(raw.get("priority") or ""), default=cfg.tasknotes.default_priority
        ),
        "due": _check_due(str(raw.get("due") or "")),
        "scheduled": _check_scheduled(str(raw.get("scheduled") or "")),
        "assignee": _canonicalize_assignee(str(raw.get("assignee") or ""), cfg),
        "project": project,
        "contexts": contexts,
        "time_estimate": _check_estimate(raw.get("time_estimate")),
        "blocked_by": blocked,
        "source": source[:240],
        "source_note": source_note,
        "selected": bool(selected) and str(selected).casefold() not in {"false", "0", "no"},
    }


def parse_document(text: str) -> dict[str, Any]:
    """Accept a JSON list or an object with ``items``, optional title and boards."""
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"items are not valid JSON: {exc}") from exc
    if isinstance(data, list):
        return {"title": "", "boards": "", "items": data}
    if isinstance(data, dict) and isinstance(data.get("items"), list):
        return {
            "title": str(data.get("title") or ""),
            "boards": str(data.get("boards") or ""),
            "items": data["items"],
        }
    raise ValueError("items JSON must be a list or an object with an 'items' list")


def _ensure_ids(items: list[dict[str, Any]]) -> None:
    used = {str(item.get("id")).strip() for item in items if str(item.get("id") or "").strip()}
    number = 1
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        current = str(item.get("id") or "").strip()
        if not current:
            while str(number) in used:
                number += 1
            current = str(number)
            used.add(current)
            number += 1
        if current in seen:
            raise ValueError(f"duplicate item id {current}")
        seen.add(current)
        item["id"] = current


def _absorb_unlabeled(token: str, fields: dict[str, str]) -> None:
    if _TIME_RE.match(token) and "scheduled" not in fields:
        fields["scheduled"] = token
        return
    if _DATE_RE.match(token) and "due" not in fields:
        fields["due"] = token
        return
    if token.casefold() in _PRIORITIES and "priority" not in fields:
        fields["priority"] = token
        return
    if token.casefold() in _STATUSES and "status" not in fields:
        fields["status"] = token
        return
    if "assignee" not in fields:
        fields["assignee"] = token


def _parse_rest(rest: str) -> tuple[str, dict[str, str]]:
    parts = [part.strip() for part in rest.split(" · ")]
    title = parts[0] if parts else ""
    fields: dict[str, str] = {}
    for part in parts[1:]:
        if not part:
            continue
        key, sep, value = part.partition(" ")
        if key in _LABELS and sep:
            fields[key] = value.strip()
        elif key in _LABELS:
            fields[key] = ""
        else:
            _absorb_unlabeled(part, fields)
    return title, fields


def parse_checklist(markdown: str) -> list[dict[str, Any]]:
    """Read the human checklist above the ``tasknotes`` fence."""
    head = markdown.split("```tasknotes", 1)[0]
    lines = head.splitlines()
    items: list[dict[str, Any]] = []
    index = 0
    while index < len(lines):
        match = _CHECK_RE.match(lines[index].rstrip())
        index += 1
        if match is None:
            continue
        title, fields = _parse_rest(match.group("rest").strip())
        detail_lines: list[str] = []
        link = ""
        while index < len(lines) and (
            lines[index].startswith("  ") or lines[index].startswith("\t")
        ):
            text = lines[index].strip()
            index += 1
            if text and _WIKI_ONLY_RE.fullmatch(text):
                link = text[2:-2]
                continue
            if text:
                detail_lines.append(text)
        item: dict[str, Any] = {
            "id": match.group("id").strip(),
            "selected": match.group("mark").lower() == "x",
            "title": title,
            "assignee": fields.get("assignee", ""),
            "due": fields.get("due", ""),
            "scheduled": fields.get("scheduled", ""),
            "project": fields.get("project", ""),
            "priority": fields.get("priority", ""),
            "status": fields.get("status", ""),
            "time_estimate": fields.get("estimate", ""),
        }
        if detail_lines:
            item["details"] = "\n".join(detail_lines)
        if link:
            item["result_path"] = link if link.endswith(".md") else f"{link}.md"
        items.append(item)
    return items


def _yaml_items(markdown: str) -> list[dict[str, Any]]:
    match = _FENCE_RE.search(markdown)
    if match is None:
        return []
    try:
        loaded = yaml.safe_load(match.group(1)) or {}
    except yaml.YAMLError:
        return []
    if isinstance(loaded, dict) and isinstance(loaded.get("items"), list):
        return [item for item in loaded["items"] if isinstance(item, dict)]
    return []


def _merge_checklist(
    yaml_items: list[dict[str, Any]], human: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    by_id = {str(item.get("id")): dict(item) for item in yaml_items}
    overlay_keys = (
        "selected",
        "title",
        "assignee",
        "due",
        "scheduled",
        "project",
        "priority",
        "status",
        "time_estimate",
    )
    merged: list[dict[str, Any]] = []
    for human_item in human:
        base = dict(by_id.get(str(human_item.get("id")), {}))
        for key in overlay_keys:
            if key in human_item:
                base[key] = human_item[key]
        if "details" in human_item:
            base["details"] = human_item["details"]
        base["id"] = human_item["id"]
        merged.append(base)
    return merged


def _wiki(path: str) -> str:
    stem = path[:-3] if path.endswith(".md") else path
    stem = stem[:-5] if stem.endswith(".base") else stem
    return f"[[{stem}]]"


def _wiki_name(path: str) -> str:
    """Filename wikilink, which is what TaskNotes shows on a card."""
    base = path.rsplit("/", 1)[-1]
    if base.endswith(".md"):
        base = base[:-3]
    elif base.endswith(".base"):
        base = base[:-5]
    return f"[[{base}]]"


def _checklist_line(item: dict[str, Any]) -> list[str]:
    mark = "x" if item.get("selected", True) else " "
    parts = [str(item.get("title") or "").strip() or "(untitled)"]
    if item.get("assignee"):
        parts.append(f"assignee {item['assignee']}")
    if item.get("due"):
        parts.append(f"due {item['due']}")
    if item.get("scheduled"):
        parts.append(f"scheduled {item['scheduled']}")
    if item.get("project"):
        parts.append(f"project {item['project']}")
    priority = str(item.get("priority") or "")
    if priority and priority != "normal":
        parts.append(f"priority {priority}")
    status = str(item.get("status") or "")
    if status and status != "open":
        parts.append(f"status {status}")
    if item.get("time_estimate"):
        parts.append(f"estimate {item['time_estimate']}")
    lines = [f"- [{mark}] `{item.get('id')}` " + " · ".join(parts)]
    details = str(item.get("details") or "").strip()
    if details:
        for line in details.splitlines():
            lines.append(f"  {line}")
    result = str(item.get("result_path") or "").strip()
    if result:
        lines.append(f"  {_wiki(result)}")
    return lines


def _dump_items(items: list[dict[str, Any]]) -> str:
    kept: list[dict[str, Any]] = []
    for item in items:
        row: dict[str, Any] = {
            "id": item.get("id"),
            "title": item.get("title") or "",
            "selected": bool(item.get("selected", True)),
        }
        for key in (
            "details",
            "status",
            "priority",
            "due",
            "scheduled",
            "assignee",
            "project",
            "source",
            "source_note",
        ):
            if item.get(key):
                row[key] = item[key]
        if item.get("contexts"):
            row["contexts"] = list(item["contexts"])
        if item.get("time_estimate"):
            row["time_estimate"] = item["time_estimate"]
        if item.get("blocked_by"):
            row["blocked_by"] = list(item["blocked_by"])
        kept.append(row)
    dumped = yaml.dump(
        {"items": kept},
        Dumper=_Dumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    ).rstrip()
    return str(dumped)


def render_proposal(
    *,
    title: str,
    items: list[dict[str, Any]],
    proposal_id: str,
    status: str,
    boards: str,
    tz_name: str,
    lead: str,
) -> str:
    meta: dict[str, Any] = {
        "pawn": "tasknotes-proposal",
        "proposal_id": proposal_id,
        "proposal_status": status,
        "title": title,
    }
    if boards:
        meta["boards"] = boards
    body_lines = [f"# {title}", "", lead.rstrip(), ""]
    for item in items:
        body_lines.extend(_checklist_line(item))
        body_lines.append("")
    body_lines.append("```tasknotes")
    body_lines.append(_dump_items(items))
    body_lines.append("```")
    body_lines.append("")
    return dump_frontmatter(meta, "\n".join(body_lines))


def _prepare_items(
    raw_items: list[Any],
    cfg: AgentConfig,
    *,
    keep_errors: bool = False,
) -> tuple[list[dict[str, Any]], list[str]]:
    dicts = [item for item in raw_items if isinstance(item, dict)]
    ignored = len(raw_items) - len(dicts)
    _ensure_ids(dicts)
    if len(dicts) > int(cfg.tasknotes.max_items or 40):
        raise ValueError(
            f"too many items ({len(dicts)}). The cap is {cfg.tasknotes.max_items}. "
            "Split the list, or raise tasknotes.max_items."
        )
    ready: list[dict[str, Any]] = []
    errors: list[str] = []
    if ignored:
        errors.append(f"{ignored} item(s) were not objects and were ignored")
    for raw in dicts:
        try:
            ready.append(normalize_item(raw, cfg))
        except ValueError as exc:
            errors.append(f"{raw.get('id', '?')}: {exc}")
            if keep_errors:
                ready.append(
                    {
                        "id": str(raw.get("id") or "?"),
                        "title": str(raw.get("title") or "(invalid)"),
                        "details": str(raw.get("details") or ""),
                        "assignee": str(raw.get("assignee") or ""),
                        "due": str(raw.get("due") or ""),
                        "scheduled": str(raw.get("scheduled") or ""),
                        "project": str(raw.get("project") or ""),
                        "selected": bool(raw.get("selected", True))
                        and str(raw.get("selected", True)).casefold() not in {"false", "0", "no"},
                        "outcome": "error",
                    }
                )
    return ready, errors


class _Existing:
    def __init__(self, key: str, meta: dict[str, Any], *, writable: bool) -> None:
        self.key = key
        self.meta = meta
        self.writable = writable

    @property
    def title(self) -> str:
        return str(self.meta.get("title") or self.key.rsplit("/", 1)[-1].removesuffix(".md"))

    @property
    def assignee(self) -> str:
        return str(self.meta.get("assignee") or "")

    @property
    def pawn_id(self) -> str:
        return str(self.meta.get("pawn_id") or "")


def _tag_matches(meta: dict[str, Any], ident: str) -> bool:
    if meta.get("pawn_id"):
        return True
    tags = meta.get("tags")
    if isinstance(tags, str):
        tags = [tags]
    if not isinstance(tags, list):
        return False
    needle = ident.casefold()
    return any(str(tag).strip().lstrip("#").casefold() == needle for tag in tags)


def _load_existing(store: Any, cfg: AgentConfig) -> tuple[list[_Existing], list[str]]:
    ident = _ident_tag(cfg)
    dirs = _dirs(cfg)
    notes: list[_Existing] = []
    warnings: list[str] = []
    scans: list[tuple[str, bool]] = [(dirs["tasks"], True)]
    external = (cfg.tasknotes.external_tasks_dir or "").strip()
    if external:
        try:
            scans.append((_safe_key(external), False))
        except ValueError as exc:
            warnings.append(str(exc))
    seen: set[str] = set()
    for folder, writable in scans:
        try:
            keys = store.list(folder, suffix=".md")
        except VaultError as exc:
            warnings.append(f"could not list {folder}: {exc}")
            continue
        for key in keys:
            if key in seen:
                continue
            seen.add(key)
            try:
                meta, _body = parse_frontmatter(store.read(key))
            except VaultError:
                continue
            if not _tag_matches(meta, ident):
                continue
            notes.append(
                _Existing(key, meta, writable=writable and is_under_agent_root(key, _root(cfg)))
            )
    return notes, warnings


def _index_fingerprints(notes: list[_Existing]) -> dict[str, list[_Existing]]:
    found: dict[str, list[_Existing]] = {}
    for note in notes:
        fp = str(note.meta.get("pawn_fingerprint") or "")
        if not fp:
            fp = _fingerprint(note.title, note.assignee)
        found.setdefault(fp, []).append(note)
    return found


def _used_stems(notes: list[_Existing], store: Any, projects_dir: str) -> set[str]:
    stems = {note.key.rsplit("/", 1)[-1].removesuffix(".md") for note in notes}
    try:
        for key in store.list(projects_dir, suffix=".md"):
            stems.add(key.rsplit("/", 1)[-1].removesuffix(".md"))
    except VaultError:
        pass
    return stems


def _unique_stem(title: str, used: set[str]) -> str:
    base = _safe_stem(title)
    if base not in used:
        used.add(base)
        return base
    number = 2
    while f"{base} {number}" in used:
        number += 1
    stem = f"{base} {number}"
    used.add(stem)
    return stem


def _task_body(item: dict[str, Any]) -> str:
    parts: list[str] = []
    details = str(item.get("details") or "").strip()
    if details:
        parts.append(details)
    source_note = str(item.get("source_note") or "").strip()
    source = str(item.get("source") or "").strip()
    if source_note:
        parts.append(f"Source: {_wiki(source_note)}")
    elif source:
        parts.append(f"Source: {source}")
    if not parts:
        return ""
    return "\n\n".join(parts) + "\n"


def _task_meta(
    item: dict[str, Any],
    *,
    ident: str,
    pawn_id: str,
    fingerprint: str,
    moment: datetime,
    blocked: list[str],
    project_stem: str,
) -> dict[str, Any]:
    meta: dict[str, Any] = {
        "tags": [ident],
        "title": item["title"],
        "status": item["status"],
        "priority": item["priority"],
    }
    if item.get("due"):
        meta["due"] = item["due"]
    if item.get("scheduled"):
        meta["scheduled"] = item["scheduled"]
    if item.get("assignee"):
        meta["assignee"] = item["assignee"]
    if item.get("project"):
        meta["project"] = item["project"]
    if project_stem:
        meta["projects"] = [f"[[{project_stem}]]"]
    if item.get("contexts"):
        meta["contexts"] = list(item["contexts"])
    if item.get("time_estimate"):
        meta["timeEstimate"] = int(item["time_estimate"])
    if blocked:
        meta["blockedBy"] = blocked
    meta["dateCreated"] = _stamp(moment)
    meta["dateModified"] = _stamp(moment)
    if item["status"] == "done":
        meta["completedDate"] = _today(moment)
    meta["pawn_id"] = pawn_id
    meta["pawn_fingerprint"] = fingerprint
    if item.get("source"):
        meta["pawn_source"] = item["source"]
    return meta


def _ensure_project_note(
    store: Any,
    *,
    projects_dir: str,
    project: str,
    used: set[str],
) -> str:
    """Return the wikilink stem for *project*, creating a stub when missing."""
    if not project:
        return ""
    stem = _safe_stem(project)
    if stem not in used:
        key = f"{projects_dir}/{stem}.md"
        body = dump_frontmatter(
            {"tags": ["project"], "title": project},
            f"# {project}\n\nTasks link here from their `projects` field.\n",
        )
        store.write(key, body)
        used.add(stem)
    return stem


def _apply_selection(items: list[dict[str, Any]], *, pick: str, take_all: bool) -> list[str]:
    if take_all:
        for item in items:
            item["selected"] = True
        return []
    wanted = {part for part in re.split(r"[\s,]+", pick.strip()) if part} if pick.strip() else set()
    if not wanted:
        return []
    known = {item["id"] for item in items}
    for item in items:
        item["selected"] = item["id"] in wanted
    return sorted(wanted - known)


def _load_proposal(store: Any, cfg: AgentConfig, path: str) -> tuple[str, dict[str, Any], str]:
    key = _under_root(cfg, path, what="proposal")
    try:
        text = store.read(key)
    except VaultError as exc:
        raise ValueError(f"could not read proposal {key}: {exc}") from exc
    meta, body = parse_frontmatter(text)
    if meta.get("pawn") != "tasknotes-proposal":
        raise ValueError(f"{key} is not a TaskNotes pick-list")
    human = parse_checklist(body)
    yaml_items = _yaml_items(body)
    note = ""
    if human:
        raw_items = _merge_checklist(yaml_items, human)
    elif yaml_items:
        raw_items = yaml_items
        note = "Could not read the checklist, so the saved item block was used."
    else:
        raise ValueError(f"{key} has no tasks on the checklist")
    return (
        key,
        {
            "title": str(meta.get("title") or "Tasks"),
            "boards": str(meta.get("boards") or ""),
            "items": raw_items,
            "proposal_id": str(meta.get("proposal_id") or ""),
        },
        note,
    )


def _latest_proposal_key(store: Any, cfg: AgentConfig) -> str:
    folder = _dirs(cfg)["proposals"]
    try:
        keys = store.list(folder, suffix=".md")
    except VaultError as exc:
        raise ValueError(f"could not list proposals: {exc}") from exc
    for key in sorted(keys, reverse=True):
        try:
            meta, _body = parse_frontmatter(store.read(key))
        except VaultError:
            continue
        if meta.get("pawn") == "tasknotes-proposal" and meta.get("proposal_status") == "open":
            return str(key)
    raise ValueError("no open pick-list. Pass --proposal, or create one with tasknotes_propose.")


def _group_label(item: dict[str, Any], mine: str) -> str:
    return str(item.get("assignee") or "") or "Unassigned"


def _format_people(items: list[dict[str, Any]], mine: str) -> list[str]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for item in items:
        groups.setdefault(_group_label(item, mine), []).append(item)

    def sort_key(label: str) -> tuple[int, str]:
        if label == mine:
            return (0, label.casefold())
        if label == "Unassigned":
            return (2, label)
        return (1, label.casefold())

    lines: list[str] = []
    for label in sorted(groups, key=sort_key):
        rows = groups[label]
        lines.append(f"{label} ({len(rows)})")
        for item in rows:
            bits = [f"`{item.get('id')}` {item.get('title')}"]
            if item.get("due"):
                bits.append(f"due {item['due']}")
            elif item.get("outcome") == "created" or item.get("status"):
                if not item.get("scheduled"):
                    bits.append("no date")
            if item.get("scheduled"):
                bits.append(f"scheduled {item['scheduled']}")
            if item.get("project"):
                bits.append(str(item["project"]))
            if item.get("result_path"):
                bits.append(_wiki(str(item["result_path"])))
            if item.get("writable") is False:
                bits.append("read-only")
            lines.append(f"- {_row_prefix(item)} · " + " · ".join(bits))
        lines.append("")
    return lines


def _row_prefix(item: dict[str, Any]) -> str:
    outcome = str(item.get("outcome") or "")
    if outcome == "created":
        return "created"
    if outcome == "skipped":
        return "already there"
    if outcome == "error":
        return "needs a fix"
    if item.get("status") and "selected" in item and item.get("result_path"):
        return str(item["status"])
    if item.get("selected", True):
        return "checked"
    return "unchecked"


def _proposal_lead(tz_name: str, items: list[dict[str, Any]]) -> str:
    created = [item for item in items if item.get("outcome") == "created"]
    held = [item for item in items if not item.get("selected", True)]
    if not created and not any(item.get("outcome") for item in items):
        return _PENDING_INTRO.format(tz=tz_name)
    bits = []
    if created:
        bits.append(f"Created {len(created)}.")
    skipped = [item for item in items if item.get("outcome") == "skipped"]
    if skipped:
        bits.append(f"Left {len(skipped)} already on the board.")
    if held:
        bits.append(f"{len(held)} still unchecked.")
    bits.append(f"Times are in {tz_name}.")
    bits.append(
        "Open a linked note in Obsidian to edit the task. Checked lines that failed stay on this list."
    )
    return " ".join(bits)


def tasknotes_propose_impl(
    cfg: AgentConfig,
    *,
    document: str,
    title: str = "",
    boards: str = "",
    now: datetime | None = None,
) -> str:
    """Write a pick-list and do not create tasks."""
    try:
        payload = parse_document(document)
        mode = _boards_mode(boards or payload.get("boards") or "")
        items, errors = _prepare_items(list(payload["items"]), cfg)
    except ValueError as exc:
        return f"Error: {exc}"
    if not items:
        detail = "; ".join(errors) if errors else "no items"
        return f"Error: nothing to propose ({detail})"
    heading = (title or payload.get("title") or "Tasks").strip() or "Tasks"
    moment = _now(cfg, now)
    proposal_id = moment.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%S")
    stem = _safe_stem(heading)[:40].rstrip() or "tasks"
    try:
        folder = _dirs(cfg)["proposals"]
        ident_ok = _ident_tag(cfg)
    except ValueError as exc:
        return f"Error: {exc}"
    del ident_ok
    key = f"{folder}/{proposal_id}-{stem}.md"
    text = render_proposal(
        title=heading,
        items=items,
        proposal_id=proposal_id,
        status="open",
        boards=mode,
        tz_name=_tz_name(cfg),
        lead=_PENDING_INTRO.format(tz=_tz_name(cfg)),
    )
    try:
        store = vault_store_from_config(cfg)
        store.write(key, text)
    except (VaultError, ValueError) as exc:
        return f"Error: {exc}"
    lines = [
        f"Pick-list written. No tasks created yet. {_wiki(key)}",
        f"Times in {_tz_name(cfg)}. Reply with the numbers to create, or ask to create the checked ones.",
        "Uncheck a line in the note to skip it. Edit assignee, due, scheduled, or project on that line.",
        "",
    ]
    lines.extend(_format_people(items, _display_name(cfg)))
    if mode:
        lines.append(f"When this list is created, boards: {mode}.")
    if errors:
        lines.append("Needs a fix before those rows can be included:")
        lines.extend(f"- {err}" for err in errors)
    return "\n".join(lines).rstrip() + "\n"


def _write_board(
    store: Any,
    cfg: AgentConfig,
    *,
    name: str,
    assignee: str,
    project: str,
    group_by: str,
    swimlane: str,
    file_stem: str = "",
) -> str:
    """Create or refresh one board. Return a one-line result."""
    ident = _ident_tag(cfg)
    views = _dirs(cfg)["views"]
    stem = file_stem or _safe_stem(name)
    key = f"{views}/{stem}.base"
    try:
        existing = store.read(key)
    except VaultError:
        existing = ""
    if existing and not existing.lstrip().startswith(_BOARD_MARKER):
        return f"- {name} — left `{key}` as-is (the pawn marker was removed, so hand edits stay)"
    group = group_by if group_by in {"status", "assignee", "priority"} else "status"
    lane = swimlane if swimlane in {"assignee", "priority"} and swimlane != group else ""
    filters: list[Any] = [f'file.hasTag("{ident}")']
    if assignee:
        filters.append(f'assignee == "{_filter_value(assignee)}"')
    if project:
        filters.append(f'project == "{_filter_value(project)}"')
    if not assignee and not project:
        filters.append('status != "done"')
    config: dict[str, Any] = {
        "columnWidth": 280,
        "hideEmptyColumns": False,
        "pinnedColumns": "open, in-progress, done",
    }
    if lane:
        config["swimLane"] = lane
    week = int(cfg.tasknotes.week_starts_on)
    if week < 0 or week > 6:
        week = 1
    document: dict[str, Any] = {
        "filters": {"and": filters},
        "views": [
            {
                "type": "tasknotesKanban",
                "name": name,
                "order": [
                    "title",
                    "status",
                    "priority",
                    "due",
                    "scheduled",
                    "assignee",
                    "project",
                    "projects",
                    "file.name",
                ],
                "groupBy": {"property": group, "direction": "ASC"},
                "config": config,
            },
            {
                "type": "tasknotesCalendar",
                "name": "Calendar",
                "dateProperty": "scheduled",
                "options": {
                    "showScheduled": True,
                    "showDue": True,
                    "showRecurring": False,
                    "showTimeEntries": False,
                    "showTimeblocks": False,
                    "showPropertyBasedEvents": False,
                    "calendarView": "timeGridWeek",
                    "firstDay": week,
                },
            },
        ],
    }
    dumped = yaml.dump(
        document,
        Dumper=_Dumper,
        default_flow_style=False,
        allow_unicode=True,
        sort_keys=False,
    )
    header = (
        f"{_BOARD_MARKER}\n"
        "# Open this file in Obsidian for the kanban and the calendar.\n"
        f"# These tasks also show on TaskNotes' own views because they are tagged {ident}.\n"
        "# Delete the first line if you customize this file. Pawn will then leave it alone.\n"
    )
    store.write(key, header + dumped, content_type="text/yaml; charset=utf-8")
    who = ", ".join(bit for bit in (assignee, project) if bit)
    scope = f" ({who})" if who else " (open tasks)"
    return f"- {name}{scope} — open `{key}` in Obsidian"


def _filter_value(value: str) -> str:
    if "\n" in value or '"' in value:
        raise ValueError(f"board filter value cannot contain quotes or newlines: {value!r}")
    return value.replace("\\", "\\\\")


def tasknotes_board_impl(
    cfg: AgentConfig,
    *,
    name: str,
    assignee: str = "",
    project: str = "",
    group_by: str = "status",
    swimlane: str = "",
) -> str:
    """Write one kanban + calendar board."""
    title = " ".join((name or "").split())
    if not title:
        return "Error: board name is empty"
    try:
        person = _canonicalize_assignee(assignee, cfg) if assignee else ""
        inferred = person_from_board_name(title)
        if inferred:
            person = person or _canonicalize_assignee(inferred, cfg)
            title = person or inferred
        proj = " ".join(project.split())
        _ident_tag(cfg)
        store = vault_store_from_config(cfg)
        line = _write_board(
            store,
            cfg,
            name=title,
            assignee=person,
            project=proj,
            group_by=group_by or "status",
            swimlane=swimlane or "",
        )
    except (VaultError, ValueError) as exc:
        return f"Error: {exc}"
    return (
        "Board ready. Tasks with this assignee or project also appear on the built-in "
        "TaskNotes kanban and calendar.\n"
        f"{line}\n"
    )


def _boards_for(
    store: Any,
    cfg: AgentConfig,
    items: list[dict[str, Any]],
    mode: str,
) -> list[str]:
    if not mode:
        return []
    lines: list[str] = []
    selected = [item for item in items if item.get("selected") and item.get("outcome") != "error"]
    used_stems: set[str] = set()
    if mode in {"assignee", "both"}:
        people = []
        seen: set[str] = set()
        for item in selected:
            person = str(item.get("assignee") or "").strip()
            if not person or person.casefold() in seen:
                continue
            seen.add(person.casefold())
            people.append(person)
        for person in people:
            lines.append(
                _write_board(
                    store,
                    cfg,
                    name=person,
                    assignee=person,
                    project="",
                    group_by="status",
                    swimlane="",
                    file_stem=_claim_stem(person, used_stems),
                )
            )
        missing = [item for item in selected if not str(item.get("assignee") or "").strip()]
        if missing:
            lines.append(
                f"- {len(missing)} task(s) have no assignee, so they stay on the main TaskNotes list only."
            )
    if mode in {"project", "both"}:
        projects: list[str] = []
        seen_p: set[str] = set()
        for item in selected:
            project = str(item.get("project") or "").strip()
            if not project or project.casefold() in seen_p:
                continue
            seen_p.add(project.casefold())
            projects.append(project)
        for project in projects:
            lines.append(
                _write_board(
                    store,
                    cfg,
                    name=project,
                    assignee="",
                    project=project,
                    group_by="status",
                    swimlane="assignee" if mode == "both" else "",
                    file_stem=_claim_stem(f"{project} project", used_stems),
                )
            )
    return lines


def _claim_stem(name: str, used: set[str]) -> str:
    stem = _safe_stem(name)
    if stem not in used:
        used.add(stem)
        return stem
    number = 2
    while f"{stem} {number}" in used:
        number += 1
    taken = f"{stem} {number}"
    used.add(taken)
    return taken


def tasknotes_commit_impl(
    cfg: AgentConfig,
    *,
    proposal: str = "",
    document: str = "",
    pick: str = "",
    take_all: bool = False,
    force: bool = False,
    boards: str = "",
    now: datetime | None = None,
) -> str:
    """Create checked tasks, or an explicit JSON list."""
    if proposal and document:
        return "Error: pass a proposal or items, not both"
    note = ""
    proposal_key = ""
    proposal_id = ""
    heading = "Tasks"
    try:
        store = vault_store_from_config(cfg)
        if document:
            payload = parse_document(document)
            raw_items = list(payload["items"])
            heading = "Tasks"
            stored_boards = str(payload.get("boards") or "")
        else:
            proposal_key = _under_root(cfg, proposal, what="proposal") if proposal else ""
            if not proposal_key:
                proposal_key = _latest_proposal_key(store, cfg)
                note = f"Using the latest open pick-list {_wiki(proposal_key)}."
            proposal_key, payload, checklist_note = _load_proposal(store, cfg, proposal_key)
            raw_items = list(payload["items"])
            heading = str(payload.get("title") or "Tasks")
            stored_boards = str(payload.get("boards") or "")
            proposal_id = str(payload.get("proposal_id") or "")
            if checklist_note:
                note = " ".join(bit for bit in (note, checklist_note) if bit)
        mode = _boards_mode(boards or stored_boards)
        _ensure_ids([item for item in raw_items if isinstance(item, dict)])
        missing = _apply_selection(
            [item for item in raw_items if isinstance(item, dict)],
            pick=pick,
            take_all=take_all,
        )
        items, errors = _prepare_items(raw_items, cfg, keep_errors=True)
    except ValueError as exc:
        return f"Error: {exc}"

    if missing:
        errors.append("no item " + ", ".join(missing))
    selected = [item for item in items if item.get("selected")]
    if not selected and not errors:
        return (
            "Nothing to create. Every line is unchecked. "
            "Check the ones you want, reply with their numbers, or ask for all of them.\n"
        )
    moment = _now(cfg, now)
    try:
        ident = _ident_tag(cfg)
        dirs = _dirs(cfg)
        existing, warnings = _load_existing(store, cfg)
    except (VaultError, ValueError) as exc:
        return f"Error: {exc}"
    fingerprints = _index_fingerprints(existing)
    used = _used_stems(existing, store, dirs["projects"])
    project_stems: dict[str, str] = {}

    creatable = [item for item in selected if item.get("outcome") != "error"]
    seen_batch: dict[str, dict[str, Any]] = {}
    for item in creatable:
        fp = _fingerprint(item["title"], item["assignee"])
        item["fingerprint"] = fp
        if fp in seen_batch and not force:
            item["outcome"] = "skipped"
            item["duplicate_of"] = fp
            continue
        matches = fingerprints.get(fp) or []
        if matches and not force:
            item["outcome"] = "skipped"
            item["result_path"] = matches[0].key
            continue
        item["outcome"] = "created"
        seen_batch[fp] = item

    for item in creatable:
        project = str(item.get("project") or "")
        if item.get("outcome") != "created" or not project or project in project_stems:
            continue
        try:
            project_stems[project] = _ensure_project_note(
                store, projects_dir=dirs["projects"], project=project, used=used
            )
        except (VaultError, ValueError) as exc:
            errors.append(f"{item['id']}: project note: {exc}")
            project_stems[project] = ""

    for item in creatable:
        if item.get("outcome") != "created":
            continue
        stem = _unique_stem(item["title"], used)
        key = f"{dirs['tasks']}/{stem}.md"
        item["stem"] = stem
        item["result_path"] = key
    for item in creatable:
        owner = seen_batch.get(str(item.get("duplicate_of") or ""))
        if owner is not None and owner.get("result_path"):
            item["result_path"] = owner["result_path"]

    for item in selected:
        if item.get("outcome") != "created":
            continue
        blocked: list[str] = []
        for dep in item.get("blocked_by") or []:
            owner = next(
                (other for other in selected if other["id"] == dep and other.get("result_path")),
                None,
            )
            if owner is None:
                item.setdefault("warnings", []).append(
                    f"dropped dependency {dep} because that item was not created"
                )
                continue
            blocked.append(_wiki_name(str(owner["result_path"])))
        meta = _task_meta(
            item,
            ident=ident,
            pawn_id=str(uuid.uuid4()),
            fingerprint=str(item["fingerprint"]),
            moment=moment,
            blocked=blocked,
            project_stem=project_stems.get(str(item.get("project") or ""), ""),
        )
        try:
            store.write(str(item["result_path"]), dump_frontmatter(meta, _task_body(item)))
        except (VaultError, ValueError) as exc:
            item["outcome"] = "error"
            errors.append(f"{item['id']}: {exc}")

    board_lines: list[str] = []
    try:
        board_lines = _boards_for(store, cfg, selected, mode)
    except (VaultError, ValueError) as exc:
        errors.append(f"boards: {exc}")

    held = [item for item in items if not item.get("selected")]
    status = "open" if held or errors else "committed"
    if proposal_key:
        try:
            rewritten = render_proposal(
                title=heading,
                items=items,
                proposal_id=proposal_id or proposal_key.rsplit("/", 1)[-1].removesuffix(".md"),
                status=status,
                boards=mode,
                tz_name=_tz_name(cfg),
                lead=_proposal_lead(_tz_name(cfg), items),
            )
            store.write(proposal_key, rewritten)
        except (VaultError, ValueError) as exc:
            errors.append(f"could not update the pick-list: {exc}")

    return _commit_report(
        cfg,
        items=items,
        errors=errors,
        warnings=warnings,
        note=note,
        board_lines=board_lines,
        proposal_key=proposal_key,
        moment=moment,
    )


def _commit_report(
    cfg: AgentConfig,
    *,
    items: list[dict[str, Any]],
    errors: list[str],
    warnings: list[str],
    note: str,
    board_lines: list[str],
    proposal_key: str,
    moment: datetime,
) -> str:
    created = [item for item in items if item.get("outcome") == "created"]
    skipped = [item for item in items if item.get("outcome") == "skipped"]
    held = [item for item in items if not item.get("selected")]
    today = _today(moment)
    overdue = [
        item
        for item in created
        if item.get("due") and str(item["due"]) < today and item.get("status") != "done"
    ]
    undated = [item for item in created if not item.get("due") and not item.get("scheduled")]
    lines = [
        f"Created {len(created)}. Skipped {len(skipped)} already on the board. "
        f"Left unchecked {len(held)}.",
        f"Times in {_tz_name(cfg)}. These notes are tagged `{_ident_tag(cfg)}`, "
        "so they show on the TaskNotes kanban and calendar after sync.",
    ]
    if note:
        lines.append(note)
    if undated:
        lines.append(
            f"No date yet ({len(undated)}): "
            + "; ".join(str(item["title"]) for item in undated)
            + ". They stay off the calendar until you set scheduled or due."
        )
    if overdue:
        lines.append(f"Already overdue: {len(overdue)}.")
    lines.append("")
    errored = [item for item in items if item.get("outcome") == "error"]
    lines.extend(_format_people([*created, *skipped, *errored, *held], _display_name(cfg)))
    dep_warnings = [f"{item['id']}: {msg}" for item in items for msg in item.get("warnings") or []]
    if dep_warnings:
        lines.append("Dependencies:")
        lines.extend(f"- {msg}" for msg in dep_warnings)
    if board_lines:
        lines.append("Boards — open the .base file in Obsidian:")
        lines.extend(board_lines)
    else:
        lines.append(
            "No extra board was written. The built-in TaskNotes kanban and calendar already include these tasks."
        )
    if proposal_key:
        lines.append(f"Pick-list: {_wiki(proposal_key)}")
    if warnings:
        lines.append("Lookup notes:")
        lines.extend(f"- {msg}" for msg in warnings)
    if errors:
        lines.append("Needs a fix:")
        lines.extend(f"- {err}" for err in errors)
    lines.append(
        "Calendar apps update only if TaskNotes export is already on "
        "(Integrations → sync trigger: scheduled). A scheduled time here is what that export sends."
    )
    return "\n".join(lines).rstrip() + "\n"


def _day(value: str) -> str:
    text = (value or "").strip()
    return text[:10] if _DATE_RE.match(text[:10] or "") else ""


def _note_item(note: _Existing) -> dict[str, Any]:
    meta = note.meta
    projects = meta.get("projects")
    project = str(meta.get("project") or "")
    if not project and isinstance(projects, list) and projects:
        project = str(projects[0]).strip("[]")
    return {
        "id": (
            note.pawn_id[:8] if note.pawn_id else note.key.rsplit("/", 1)[-1].removesuffix(".md")
        ),
        "title": note.title,
        "assignee": note.assignee,
        "due": str(meta.get("due") or ""),
        "scheduled": str(meta.get("scheduled") or ""),
        "project": project,
        "status": str(meta.get("status") or "open"),
        "selected": True,
        "result_path": note.key,
        "writable": note.writable,
        "priority": str(meta.get("priority") or ""),
    }


def tasknotes_list_impl(
    cfg: AgentConfig,
    *,
    assignee: str = "",
    mine: bool = False,
    project: str = "",
    status: str = "",
    include_done: bool = False,
    undated: bool = False,
    scheduled_from: str = "",
    scheduled_to: str = "",
    limit: int = 40,
) -> str:
    """List task notes Pawn can see, grouped by person."""
    try:
        if undated and (scheduled_from or scheduled_to):
            raise ValueError("use either --undated or a date range, not both")
        if scheduled_from:
            _check_due(scheduled_from)
        if scheduled_to:
            _check_due(scheduled_to)
        status_filter = _check_status(status, default="") if status else ""
        person = _canonicalize_assignee(assignee, cfg) if assignee else ""
        if mine:
            person = _display_name(cfg)
        store = vault_store_from_config(cfg)
        notes, warnings = _load_existing(store, cfg)
    except (VaultError, ValueError) as exc:
        return f"Error: {exc}"

    rows = [_note_item(note) for note in notes]
    if person:
        rows = [row for row in rows if str(row["assignee"]).casefold() == person.casefold()]
    if project:
        needle = project.casefold()
        rows = [row for row in rows if needle in str(row["project"]).casefold()]
    if status_filter:
        rows = [row for row in rows if row["status"] == status_filter]
    elif not include_done:
        rows = [row for row in rows if row["status"] != "done"]
    if undated:
        rows = [row for row in rows if not row["due"] and not row["scheduled"]]
    if scheduled_from or scheduled_to:

        def _in_range(row: dict[str, Any]) -> bool:
            days = [day for day in (_day(str(row["due"])), _day(str(row["scheduled"]))) if day]
            if not days:
                return False
            return any(
                (not scheduled_from or day >= scheduled_from)
                and (not scheduled_to or day <= scheduled_to)
                for day in days
            )

        rows = [row for row in rows if _in_range(row)]

    def _sort(row: dict[str, Any]) -> tuple[str, str, str]:
        return (
            _day(str(row["scheduled"])) or _day(str(row["due"])) or "9999-99-99",
            str(row["due"] or ""),
            str(row["title"]).casefold(),
        )

    rows.sort(key=_sort)
    total = len(rows)
    cap = max(1, int(limit))
    shown = rows[:cap]
    readonly = [row for row in shown if not row.get("writable")]
    undated_n = sum(1 for row in shown if not row["due"] and not row["scheduled"])
    lines = [
        f"{total} task(s). Showing {len(shown)}. Times in {_tz_name(cfg)}.",
    ]
    if not include_done and not status_filter:
        lines.append("Done tasks are hidden. Pass --include-done to see them.")
    if readonly:
        lines.append(
            f"{len(readonly)} shown live in the TaskNotes folder. "
            "Pawn can read those and will not rewrite them."
        )
    lines.append("")
    if not shown:
        lines.append("No matching tasks.")
    else:
        lines.extend(_format_people(shown, _display_name(cfg)))
    if undated_n:
        lines.append(f"No date yet in this list: {undated_n}.")
    if warnings:
        lines.extend(f"- {msg}" for msg in warnings)
    return "\n".join(lines).rstrip() + "\n"


def _resolve_note(notes: list[_Existing], ident: str) -> _Existing | str:
    text = ident.strip()
    if not text:
        return "Error: --id is empty"
    if "/" in text or text.endswith(".md"):
        try:
            key = _safe_key(text if text.endswith(".md") else f"{text}.md")
        except ValueError as exc:
            return f"Error: {exc}"
        for note in notes:
            if note.key == key:
                return note
        return f"Error: no task at {key}"
    folded = text.casefold()
    by_id = [
        note
        for note in notes
        if note.pawn_id and (note.pawn_id == text or note.pawn_id.startswith(text))
    ]
    if len(by_id) == 1:
        return by_id[0]
    if len(by_id) > 1:
        return "Error: id prefix matches more than one task: " + ", ".join(
            note.key for note in by_id[:5]
        )
    by_title = [note for note in notes if note.title.casefold() == folded]
    if len(by_title) == 1:
        return by_title[0]
    if len(by_title) > 1:
        return "Error: more than one task has that title. Pass the path or the id. " + ", ".join(
            note.key for note in by_title[:5]
        )
    return f"Error: no task matches {text!r}"


def tasknotes_update_impl(
    cfg: AgentConfig,
    *,
    ident: str,
    title: str | None = None,
    status: str | None = None,
    priority: str | None = None,
    assignee: str | None = None,
    due: str | None = None,
    scheduled: str | None = None,
    project: str | None = None,
    details: str | None = None,
    estimate: str | None = None,
    clear_due: bool = False,
    clear_scheduled: bool = False,
    clear_assignee: bool = False,
    clear_project: bool = False,
    clear_estimate: bool = False,
    now: datetime | None = None,
) -> str:
    """Change fields on one task. The file name stays so links keep working."""
    changes = [
        title,
        status,
        priority,
        assignee,
        due,
        scheduled,
        project,
        details,
        estimate,
    ]
    clears = [clear_due, clear_scheduled, clear_assignee, clear_project, clear_estimate]
    if all(value is None for value in changes) and not any(clears):
        return "Error: nothing to change. Pass a field or a --clear-* flag."
    if (due is not None and clear_due) or (scheduled is not None and clear_scheduled):
        return "Error: do not pass a date and clear that date in the same call"
    try:
        store = vault_store_from_config(cfg)
        notes, _warnings = _load_existing(store, cfg)
        found = _resolve_note(notes, ident)
        if isinstance(found, str):
            return found
        if not found.writable or not is_under_agent_root(found.key, _root(cfg)):
            return (
                f"Error: {found.key} is outside {_root(cfg)}/ so it was not changed. "
                "Edit that task in Obsidian."
            )
        meta, body = parse_frontmatter(store.read(found.key))
        moment = _now(cfg, now)
        if title is not None:
            meta["title"] = _clean_title(title)
        if status is not None:
            meta["status"] = _check_status(status, default=cfg.tasknotes.default_status)
        if priority is not None:
            meta["priority"] = _check_priority(priority, default=cfg.tasknotes.default_priority)
        if clear_assignee:
            meta.pop("assignee", None)
        elif assignee is not None:
            person = _canonicalize_assignee(assignee, cfg)
            if person:
                meta["assignee"] = person
            else:
                meta.pop("assignee", None)
        if clear_due:
            meta.pop("due", None)
        elif due is not None:
            checked = _check_due(due)
            if checked:
                meta["due"] = checked
            else:
                meta.pop("due", None)
        if clear_scheduled:
            meta.pop("scheduled", None)
        elif scheduled is not None:
            checked_when = _check_scheduled(scheduled)
            if checked_when:
                meta["scheduled"] = checked_when
            else:
                meta.pop("scheduled", None)
        if clear_project:
            meta.pop("project", None)
            meta.pop("projects", None)
        elif project is not None:
            proj = " ".join(project.split())
            if proj:
                used = _used_stems(notes, store, _dirs(cfg)["projects"])
                stem = _ensure_project_note(
                    store, projects_dir=_dirs(cfg)["projects"], project=proj, used=used
                )
                meta["project"] = proj
                meta["projects"] = [f"[[{stem}]]"]
            else:
                meta.pop("project", None)
                meta.pop("projects", None)
        if clear_estimate:
            meta.pop("timeEstimate", None)
        elif estimate is not None:
            minutes = _check_estimate(estimate)
            if minutes:
                meta["timeEstimate"] = minutes
            else:
                meta.pop("timeEstimate", None)
        if meta.get("status") == "done":
            meta.setdefault("completedDate", _today(moment))
        elif status is not None:
            meta.pop("completedDate", None)
        meta["dateModified"] = _stamp(moment)
        current_title = str(meta.get("title") or found.title)
        current_assignee = str(meta.get("assignee") or "")
        meta["pawn_fingerprint"] = _fingerprint(current_title, current_assignee)
        if details is not None:
            source_lines = [line for line in body.splitlines() if line.startswith("Source:")]
            text = str(details).strip()
            if source_lines:
                text = (text + "\n\n" if text else "") + "\n".join(source_lines)
            body = text + ("\n" if text else "")
        store.write(
            found.key,
            dump_frontmatter(meta, body if body.endswith("\n") or not body else body + "\n"),
        )
    except (VaultError, ValueError) as exc:
        return f"Error: {exc}"
    file_name = found.key.rsplit("/", 1)[-1]
    return (
        f"Updated {_wiki(found.key)} ({meta.get('status')}). "
        f"File name left as {file_name} so existing links keep working. "
        "The TaskNotes card uses the title property.\n"
    )
