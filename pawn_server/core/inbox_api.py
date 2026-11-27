"""Portal Inbox routes (``/v1/portal/inbox/*``): ideas, research captures, counts.

Coworker items keep their own routes (``/v1/items*`` in ``api_server``). This
router covers the two vault-backed Inbox sections the Obsidian plugin used to
read straight from the vault:

- **Ideas**: ``Ideas/*.md`` notes with ``status: inbox`` (written by ``/idea``).
  Triage rewrites only the ``status:`` frontmatter line, and *goal* / *task*
  also write a Goals.md thread or a TaskNotes note that links back.
- **Captures**: research inbox atoms under ``capture.inbox_dir`` with status
  ``inbox`` or ``proposed``. Filing goes through ``POST /v1/captures/file``.
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, ConfigDict

from pawn_core.vault import VaultError, VaultNotFound, VaultStore, parse_frontmatter

logger = logging.getLogger(__name__)

IDEA_ACTIONS = ("keep", "goal", "task", "drop")
_IDEA_STATUS = {"keep": "later", "goal": "goal", "task": "task", "drop": "dropped"}
CAPTURE_OPEN = ("inbox", "proposed")
_UNSAFE = re.compile(r'[\\/:*?"<>|\x00-\x1f]')
_FRONT = re.compile(r"\A---\r?\n(.*?)\r?\n---", re.S)
_PARK_DO = "Link related notes and develop the idea. Do not notify."


class IdeaCreate(BaseModel):
    model_config = ConfigDict(extra="ignore")

    line: str


class IdeaAction(BaseModel):
    model_config = ConfigDict(extra="ignore")

    path: str
    action: str
    title: Optional[str] = None


# ── helpers ───────────────────────────────────────────────────────────────────


def _store(cfg: Any) -> VaultStore:
    from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

    try:
        return vault_store_from_config(cfg)
    except Exception as exc:  # noqa: BLE001 - surfaced as 503
        raise HTTPException(status_code=503, detail=f"Vault is not configured: {exc}") from exc


def _agent_root(cfg: Any) -> str:
    root = getattr(getattr(cfg, "vault", None), "agent_root", None) or "Pawn"
    return str(root).strip().strip("/") or "Pawn"


def _ideas_folder() -> str:
    from pawn_agent.tools.ideas_impl import IDEAS_FOLDER  # noqa: PLC0415

    return IDEAS_FOLDER


def _goals_path(cfg: Any) -> str:
    return str(getattr(getattr(cfg, "coworker", None), "goals_path", "") or "Goals.md")


def _tasks_dir(cfg: Any) -> str:
    try:
        from pawn_agent.tools.tasknotes_impl import _dirs  # noqa: PLC0415

        return _dirs(cfg)["tasks"]
    except Exception:  # noqa: BLE001 - config without a tasknotes section
        return f"{_agent_root(cfg)}/TaskNotes/Tasks"


def _str(meta: dict[str, Any], key: str) -> str:
    value = meta.get(key)
    if value is None:
        return ""
    return str(value).strip().strip("\"'")


def _title(body: str, fallback: str) -> str:
    for line in body.splitlines():
        if line.startswith("# "):
            return line[2:].strip() or fallback
    return fallback


def _stem(key: str) -> str:
    name = key.rsplit("/", 1)[-1]
    return name[:-3] if name.endswith(".md") else name


def idea_line(body: str) -> str:
    """The idea text: the body without its leading ``# heading``."""
    text = body.strip()
    lines = text.splitlines()
    if lines and lines[0].startswith("#"):
        return "\n".join(lines[1:]).strip()
    return text


def with_status(text: str, status: str) -> str:
    """Set ``status:`` in the frontmatter, leaving every other line alone."""
    match = _FRONT.match(text)
    if match is None:
        return f"---\ntags: [idea]\nstatus: {status}\n---\n\n{text}"
    meta = match.group(1)
    if re.search(r"^status:", meta, re.M):
        meta = re.sub(r"^status:.*$", f"status: {status}", meta, count=1, flags=re.M)
    else:
        meta = f"{meta}\nstatus: {status}"
    return f"---\n{meta}\n---{text[match.end():]}"


def _meta_of(text: str) -> tuple[dict[str, Any], str]:
    try:
        return parse_frontmatter(text)
    except Exception:  # noqa: BLE001 - a broken fence still lists by body
        return {}, text


def _created_of(store: VaultStore, key: str, meta: dict[str, Any]) -> str:
    created = _str(meta, "created") or _str(meta, "captured_at")
    if created:
        return created
    stat = store.stat(key)
    if stat is not None and getattr(stat, "last_modified", None):
        lm = stat.last_modified
        return lm.isoformat() if hasattr(lm, "isoformat") else str(lm)
    day = _stem(key)[:10]
    return day if re.match(r"\d{4}-\d{2}-\d{2}$", day) else ""


# ── ideas ─────────────────────────────────────────────────────────────────────


def list_ideas(cfg: Any, store: VaultStore, status: Optional[str] = "inbox") -> list[dict[str, Any]]:
    """Idea notes, newest first. *status* None (or "all") lists every status."""
    want = None if status in (None, "", "all", "*") else status
    out: list[dict[str, Any]] = []
    for key in store.list(_ideas_folder() + "/"):
        try:
            text = store.read(key)
        except VaultError:
            continue
        meta, body = _meta_of(text)
        st = _str(meta, "status")
        if want is not None and st != want:
            continue
        out.append(
            {
                "path": key,
                "title": _title(body, _stem(key)),
                "line": idea_line(body),
                "status": st,
                "created": _created_of(store, key, meta),
            }
        )
    out.sort(key=lambda r: (r["created"], r["path"]), reverse=True)
    return out


def _unique_key(store: VaultStore, folder: str, stem: str) -> str:
    key = f"{folder}/{stem}.md"
    n = 2
    while store.exists(key):
        key = f"{folder}/{stem} {n}.md"
        n += 1
    return key


def _task_stem(title: str) -> str:
    cleaned = re.sub(r"\s+", " ", _UNSAFE.sub(" ", title)).strip().rstrip(".")
    return cleaned[:80].strip() or "task"


def apply_idea_action(
    cfg: Any, store: VaultStore, path: str, action: str, title: Optional[str] = None
) -> dict[str, Any]:
    """Triage one idea note. Returns ``{receipt, path, written?}``."""
    from pawn_core.goals import GoalInsertError, insert_goal_thread  # noqa: PLC0415

    action = (action or "").strip().lower()
    if action not in IDEA_ACTIONS:
        raise HTTPException(status_code=422, detail=f"Unknown idea action {action!r}")
    folder = _ideas_folder()
    if not path.startswith(folder + "/") or not path.endswith(".md"):
        raise HTTPException(status_code=422, detail=f"Not an idea note: {path}")
    try:
        text = store.read(path)
    except VaultNotFound as exc:
        raise HTTPException(status_code=404, detail=f"Not found: {path}") from exc
    _meta, body = _meta_of(text)
    name = " ".join((title or _title(body, _stem(path))).split())[:120]
    line = idea_line(body)
    link = f"[[{path[:-3]}]]"
    written: Optional[str] = None

    if action == "goal":
        if not name:
            raise HTTPException(status_code=422, detail="A goal thread needs a name.")
        key = _goals_path(cfg)
        try:
            current = store.read(key)
        except VaultNotFound:
            current = None
        try:
            updated = insert_goal_thread(current, name=name, status="parked", note=link, do=_PARK_DO)
        except GoalInsertError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        store.write(key, updated, skip_guards=True)
        written = key
        receipt = f"Parked '{name}' in {key}."
    elif action == "task":
        import json  # noqa: PLC0415

        key = _unique_key(store, _tasks_dir(cfg), _task_stem(name or "task"))
        note = (
            f"---\ntags: [task]\ntitle: {json.dumps(name or 'task')}\nstatus: open\npriority: normal\n---\n\n"
            f"Source: {link}\n\n{line}\n"
        )
        store.write(key, note, skip_guards=True)
        written = key
        receipt = f"Task {key}."
    elif action == "keep":
        receipt = "Kept for later."
    else:
        receipt = "Dropped."

    store.write(path, with_status(text, _IDEA_STATUS[action]), skip_guards=True)
    return {"receipt": receipt, "path": path, "written": written, "status": _IDEA_STATUS[action]}


# ── captures ──────────────────────────────────────────────────────────────────


def _inbox_dir(cfg: Any) -> str:
    from pawn_server.core import capture_config  # noqa: PLC0415

    return capture_config.inbox_dir(cfg)


def list_capture_atoms(cfg: Any, store: VaultStore, scope: str = "inbox") -> list[dict[str, Any]]:
    """Research inbox atoms, newest first. ``scope=all`` keeps filed ones too."""
    out: list[dict[str, Any]] = []
    for key in store.list(_inbox_dir(cfg) + "/"):
        try:
            text = store.read(key)
        except VaultError:
            continue
        meta, body = _meta_of(text)
        status = _str(meta, "status")
        if scope != "all" and status not in CAPTURE_OPEN:
            continue
        out.append(
            {
                "path": key,
                "title": _title(body, _stem(key)),
                "status": status,
                "collection": _str(meta, "collection"),
                "entity": _str(meta, "entity"),
                "hint": _str(meta, "hint"),
                "type": _str(meta, "type"),
                "snippet_kind": _str(meta, "kind"),
                "source_url": _str(meta, "source_url"),
                "caption": _str(meta, "caption"),
                "created": _created_of(store, key, meta),
            }
        )
    out.sort(key=lambda r: (r["created"], r["path"]), reverse=True)
    return out


def _open_items(cfg: Any) -> int:
    try:
        from pawn_agent.core.coworker import db as itemdb  # noqa: PLC0415

        return int(itemdb.count_items(cfg.db_dsn, statuses=list(itemdb.OPEN_STATUSES)))
    except Exception as exc:  # noqa: BLE001 - counts are a badge, never fatal
        logger.debug("inbox item count failed: %s", exc)
        return 0


def counts(cfg: Any) -> dict[str, int]:
    out = {"items": _open_items(cfg), "ideas": 0, "captures": 0}
    try:
        store = _store(cfg)
    except HTTPException:
        return out
    try:
        out["ideas"] = len(list_ideas(cfg, store))
        out["captures"] = len(list_capture_atoms(cfg, store))
    except Exception as exc:  # noqa: BLE001
        logger.debug("inbox vault counts failed: %s", exc)
    return out


# ── router ────────────────────────────────────────────────────────────────────


def build_router(get_cfg: Callable[[], Any], require_token: Callable[..., None]) -> APIRouter:
    router = APIRouter(
        prefix="/v1/portal/inbox", tags=["portal"], dependencies=[Depends(require_token)]
    )

    @router.get("/counts")
    async def inbox_counts(cfg: Any = Depends(get_cfg)) -> dict:
        return await asyncio.to_thread(counts, cfg)

    @router.get("/ideas")
    async def ideas_list(status: str = Query(default="inbox"), cfg: Any = Depends(get_cfg)) -> dict:
        store = _store(cfg)
        return {"ideas": await asyncio.to_thread(list_ideas, cfg, store, status)}

    @router.post("/ideas", status_code=201)
    async def ideas_create(body: IdeaCreate, cfg: Any = Depends(get_cfg)) -> dict:
        from pawn_agent.tools.ideas_impl import capture_idea  # noqa: PLC0415

        store = _store(cfg)
        try:
            receipt = await asyncio.to_thread(capture_idea, cfg, line=body.line, store=store)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return {"receipt": receipt}

    @router.post("/ideas/action")
    async def ideas_action(body: IdeaAction, cfg: Any = Depends(get_cfg)) -> dict:
        store = _store(cfg)
        return await asyncio.to_thread(
            apply_idea_action, cfg, store, body.path, body.action, body.title
        )

    @router.get("/captures")
    async def captures_inbox(scope: str = Query(default="inbox"), cfg: Any = Depends(get_cfg)) -> dict:
        store = _store(cfg)
        return {"captures": await asyncio.to_thread(list_capture_atoms, cfg, store, scope)}

    return router
