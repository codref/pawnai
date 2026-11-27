"""Seed a new sallm session with the first turns of another (portal fork).

Only the chat transcript is copied (user/assistant ``chat`` messages). Skill
frames reset to the default skill, and Lance memory chunks are not copied:
the forked session starts with the transcript but none of the derived
long-term memory. See docs/PORTAL.md.
"""

from __future__ import annotations

from typing import Any, Iterable

from sallm.state.repository import SessionRepository

from pawn_agent.core.sallm_factory import resolve_state_dir

_CHAT_ROLES = ("user", "assistant")


def fork_session(
    cfg: Any,
    source_session: str,
    target_session: str,
    *,
    turns: int,
    fallback: Iterable[tuple[str, str]] = (),
) -> int:
    """Copy the first *turns* user/assistant messages into *target_session*.

    *fallback* is ``(role, content)`` pairs (the portal mirror) used for any
    turn sallm has no record of, e.g. a chat started before the mirror
    existed or cleared with ``/reset``. Returns the number of copied messages.
    """
    repo = SessionRepository(resolve_state_dir(cfg) / "state.db")
    try:
        source = [
            m
            for m in repo.list_messages(source_session)
            if m.kind == "chat" and m.role in _CHAT_ROLES
        ]
        pairs = [(m.role, m.content) for m in source[:turns]]
        backup = list(fallback)
        if len(pairs) < turns:
            pairs.extend(backup[len(pairs) : turns])
        repo.ensure_session(target_session)
        for role, content in pairs:
            repo.append_message(target_session, role=role, content=content, kind="chat")
        return len(pairs)
    finally:
        repo.close()
