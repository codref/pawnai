"""Read Goals.md and draft a proposal note. Direct writes stay in slash commands."""

from __future__ import annotations

from typing import Any, Optional

from pawn_agent.utils.config import AgentConfig
from pawn_core.goals import GoalInsertError, insert_goal_thread, load_goals_text
from pawn_core.vault import VaultNotFound, dump_frontmatter
from pawn_core.vault_config import vault_store_from_config

_PROPOSAL_NAME = "goal-proposal.md"


def proposal_key(cfg: AgentConfig) -> str:
    """Vault key of the latest chat-drafted goals proposal."""
    folder = (cfg.coworker.reviews_dir or "Pawn/Reviews").strip().strip("/")
    return f"{folder}/{_PROPOSAL_NAME}"


def read_note(store: Any, key: str) -> Optional[str]:
    """Return a note body, or None when the key is missing."""
    try:
        body = store.read(key)
    except VaultNotFound:
        return None
    return body if isinstance(body, str) else str(body)


def format_goals(text: Optional[str]) -> str:
    """List active and parked threads."""
    if text is None:
        return "No Goals.md yet."
    goals = load_goals_text(text)
    if not goals.valid:
        return "Goals.md is not a goals note."
    lines = ["## Active"]
    if not goals.active:
        lines.append("_None._")
    for thread in goals.active:
        lines.append(f"- {thread.name}")
        if thread.why:
            lines.append(f"  why: {thread.why}")
    lines.append("")
    lines.append("## Parked")
    if not goals.parked:
        lines.append("_None._")
    for thread in goals.parked:
        lines.append(f"- {thread.name}")
    return "\n".join(lines)


def render_goal_proposal(goals_text: str, *, name: str) -> str:
    """A review note whose ```goals fence is the full proposed Goals.md."""
    fence = goals_text.strip() + "\n"
    body = (
        "# Goal proposal\n\n"
        f"Proposed thread: {name}\n\n"
        "Goals.md was not changed. Run `/goal apply`, or open this note and "
        "use Apply goals proposal.\n\n"
        f"```goals\n{fence}```\n"
    )
    return dump_frontmatter({"pawn": "review"}, body)


def goal_propose_impl(
    cfg: AgentConfig,
    *,
    name: str,
    why: str = "",
    movement: str = "",
    interrupt: str = "",
    note: str = "",
    do: str = "",
    status: str = "active",
    store: Any = None,
) -> str:
    """Write ``Pawn/Reviews/goal-proposal.md``. Does not touch Goals.md."""
    kind = (status or "active").strip().lower()
    if kind not in {"active", "parked"}:
        raise GoalInsertError("status must be active or parked.")
    vault = store if store is not None else vault_store_from_config(cfg)
    current = read_note(vault, cfg.coworker.goals_path)
    merged = insert_goal_thread(
        current,
        name=name,
        status=kind,
        why=why,
        movement=movement,
        interrupt=interrupt,
        note=note,
        do=do,
    )
    key = proposal_key(cfg)
    title = " ".join(name.split())
    vault.write(key, render_goal_proposal(merged, name=title))
    return (
        f"Drafted a goals proposal at {key}. Goals.md was not changed. "
        "Run /goal apply, or open the note and use Apply goals proposal."
    )
