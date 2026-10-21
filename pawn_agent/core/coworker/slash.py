"""Chat slash commands for ideas, goals, and inbox triage.

``/idea``, ``/goal``, ``/park``, ``/goals``, and ``/inbox`` run here and do
not start a model turn. A normal prompt never writes Goals.md; that file
changes only from ``/goal``, ``/park``, and ``/goal apply``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Optional

from pawn_agent.utils.config import AgentConfig
from pawn_core.goals import GoalInsertError, insert_goal_thread
from pawn_core.vault import VaultError

logger = logging.getLogger(__name__)

SLASH_HELP = (
    "Supported slash commands: /idea <line>, /goal <line>, /goal apply, "
    "/park <line>, /goals, /inbox, /model [id|reset], /stats, /reset, /exit, /quit. "
    "Triage an inbox item with todo, file, task, delete, ignore, or later plus its id."
)

_IDEA_USAGE = "Usage: /idea <one line>\nExample: /idea implement multi-model in pawnai"
_GOAL_USAGE = (
    "Usage: /goal <one line>\n"
    "Writes an Active thread in Goals.md. /goal apply writes the latest proposal."
)
_PARK_USAGE = "Usage: /park <one line>\nAdds a Parked thread in Goals.md."

_PARK_DO = "Link related notes and develop the idea. Do not notify."


@dataclass(frozen=True)
class ChatResolution:
    """What a chat surface should do with one user message."""

    mode: str  # "reply" shows text; "prompt" runs the agent with text
    text: str
    rewritten: bool = False


@dataclass(frozen=True)
class _Directive:
    """Internal classification. ``kind`` is prompt, usage, direct, or triage."""

    kind: str
    text: str = ""
    command: str = ""
    arg: str = ""
    extra: str = ""


def _command_arg(text: str, command: str) -> Optional[str]:
    """Return the text after ``/command`` when the first word matches."""
    head, _, tail = text.partition(" ")
    if head.lower() != command:
        return None
    return tail.strip()


def classify_chat_text(text: str, *, triage: bool = False) -> _Directive:
    """Classify one chat message. Call :func:`resolve_chat_message` to run it."""
    raw = (text or "").strip()
    if not raw:
        return _Directive("prompt", "")

    idea = _command_arg(raw, "/idea")
    if idea is not None:
        if not idea:
            return _Directive("usage", _IDEA_USAGE)
        return _Directive("direct", command="idea", arg=idea)

    goal = _command_arg(raw, "/goal")
    if goal is not None:
        if not goal:
            return _Directive("usage", _GOAL_USAGE)
        if goal.lower() == "apply":
            return _Directive("direct", command="goal_apply")
        return _Directive("direct", command="goal", arg=goal)

    park = _command_arg(raw, "/park")
    if park is not None:
        if not park:
            return _Directive("usage", _PARK_USAGE)
        return _Directive("direct", command="park", arg=park)

    if raw.lower() == "/goals":
        return _Directive("direct", command="goals")
    if raw.lower() == "/inbox":
        return _Directive("direct", command="inbox")

    if triage:
        from pawn_agent.core.coworker.actions import parse_coworker_command  # noqa: PLC0415

        parsed = parse_coworker_command(raw)
        if parsed is not None:
            action, item_id, arg = parsed
            return _Directive("triage", command=action, arg=item_id, extra=arg or "")

    return _Directive("prompt", raw)


def _store(cfg: AgentConfig, store: Any) -> Any:
    if store is not None:
        return store
    from pawn_core.vault_config import vault_store_from_config  # noqa: PLC0415

    return vault_store_from_config(cfg)


def _write_goals(store: Any, key: str, body: str) -> None:
    store.write(key, body, skip_guards=True)


def _add_thread(cfg: AgentConfig, store: Any, line: str, *, status: str) -> str:
    from pawn_agent.tools.goals_impl import read_note  # noqa: PLC0415

    key = cfg.coworker.goals_path
    current = read_note(store, key)
    if status == "active":
        updated = insert_goal_thread(
            current,
            name=line,
            status="active",
            why=line,
        )
        where = "Active"
    else:
        updated = insert_goal_thread(
            current,
            name=line,
            status="parked",
            do=_PARK_DO,
        )
        where = "Parked"
    _write_goals(store, key, updated)
    title = " ".join(line.split())
    if status == "active":
        return (
            f"Added '{title}' under {where} in {key}. "
            "Fill movement and interrupt when this thread should drive interrupts."
        )
    return f"Parked '{title}' in {key}."


def _apply_proposal(cfg: AgentConfig, store: Any) -> str:
    from pawn_agent.core.coworker.review import extract_goals_block  # noqa: PLC0415
    from pawn_agent.tools.goals_impl import proposal_key, read_note  # noqa: PLC0415
    from pawn_core.goals import parse_goals  # noqa: PLC0415

    key = proposal_key(cfg)
    note = read_note(store, key)
    if not note:
        return f"No goal proposal at {key}."
    proposed = extract_goals_block(note)
    if not proposed.strip():
        return f"{key} has no ```goals block."
    parsed = parse_goals(proposed)
    if not parsed.valid:
        return "The proposal is not a goals note, so Goals.md was left unchanged."
    dest = cfg.coworker.goals_path
    _write_goals(store, dest, proposed if proposed.endswith("\n") else proposed + "\n")
    return f"Wrote {dest} from {key}."


def _list_goals(cfg: AgentConfig, store: Any) -> str:
    from pawn_agent.tools.goals_impl import format_goals, read_note  # noqa: PLC0415

    return format_goals(read_note(store, cfg.coworker.goals_path))


def _list_inbox(cfg: AgentConfig) -> str:
    from pawn_agent.core.coworker import db as itemdb  # noqa: PLC0415

    limit = 50
    try:
        total = itemdb.count_items(cfg.db_dsn, statuses=list(itemdb.OPEN_STATUSES))
        rows = itemdb.list_items(cfg.db_dsn, statuses=list(itemdb.OPEN_STATUSES), limit=limit)
    except Exception as exc:
        logger.warning("inbox list failed: %s", exc)
        return f"Error: could not read the inbox ({exc})"
    if not rows:
        return "Nothing open."
    lines = ["Open items:"]
    for item in rows:
        short_id = item.get("short_id") or item.get("id") or ""
        kind = item.get("kind") or "item"
        text = " ".join(str(item.get("text") or "").split())
        thread = item.get("thread") or ""
        suffix = f" ({thread})" if thread else ""
        lines.append(f"- {short_id} {kind} — {text}{suffix}")
    if total > len(rows):
        lines.append(f"… and {total - len(rows)} more.")
    return "\n".join(lines)


def _capture_idea(cfg: AgentConfig, store: Any, line: str) -> str:
    from pawn_agent.tools.ideas_impl import capture_idea  # noqa: PLC0415

    return capture_idea(cfg, line=line, store=store)


def _run_direct(cfg: AgentConfig, directive: _Directive, *, store: Any) -> str:
    if directive.command == "idea":
        return _capture_idea(cfg, store, directive.arg)
    if directive.command == "goal":
        return _add_thread(cfg, store, directive.arg, status="active")
    if directive.command == "park":
        return _add_thread(cfg, store, directive.arg, status="parked")
    if directive.command == "goal_apply":
        return _apply_proposal(cfg, store)
    if directive.command == "goals":
        return _list_goals(cfg, store)
    if directive.command == "inbox":
        return _list_inbox(cfg)
    return "Error: unknown command."


async def resolve_chat_message(
    cfg: AgentConfig,
    text: str,
    *,
    registry: Any = None,
    store: Any = None,
) -> ChatResolution:
    """Turn a user message into a reply or an agent prompt.

    Direct idea, goal, and inbox commands run here. None of them start a
    model turn.
    """
    from pawn_agent.core.vision import try_vision_command  # noqa: PLC0415
    from pawn_agent.utils.model_catalog import try_model_command  # noqa: PLC0415

    vision_reply = try_vision_command(cfg, text)
    if vision_reply is not None:
        return ChatResolution("reply", vision_reply)

    model_reply = try_model_command(cfg, text)
    if model_reply is not None:
        return ChatResolution("reply", model_reply)

    triage = bool(getattr(getattr(cfg, "coworker", None), "enabled", False))
    classified = classify_chat_text(text, triage=triage)
    if classified.kind == "prompt":
        return ChatResolution("prompt", classified.text)
    if classified.kind == "usage":
        return ChatResolution("reply", classified.text)
    if classified.kind == "triage":
        from pawn_agent.core.coworker.actions import apply_action  # noqa: PLC0415

        try:
            receipt = await apply_action(
                cfg,
                classified.arg,
                classified.command,
                classified.extra or None,
                registry=registry,
                store=store,
            )
        except Exception as exc:
            logger.exception("triage command failed")
            return ChatResolution("reply", f"Error: {exc}")
        return ChatResolution("reply", receipt)

    try:
        vault = _store(cfg, store)
        reply = _run_direct(cfg, classified, store=vault)
    except GoalInsertError as exc:
        return ChatResolution("reply", str(exc))
    except VaultError as exc:
        return ChatResolution("reply", f"Error: {exc}")
    except Exception as exc:
        logger.exception("slash command failed")
        return ChatResolution("reply", f"Error: {exc}")
    return ChatResolution("reply", reply)
