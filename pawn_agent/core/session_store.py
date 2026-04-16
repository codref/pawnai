"""Persistent storage for agent conversation session turns.

Each turn (a user prompt + assistant response pair) is stored as an
append-only row in ``agent_session_turns``, keyed by an opaque
``source_id`` (queue message_id or a uuid4 for API/chat requests).

``INSERT ... ON CONFLICT (source_id) DO NOTHING`` makes all writes
idempotent — safe for queue redeliveries and HTTP retries.

Both the API server and the queue listener call :func:`load_history`
and :func:`append_turn` so conversation context is shared across
all entry points that use the same ``session_id``.
"""

from __future__ import annotations

import json
import logging
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Generator, List

logger = logging.getLogger(__name__)

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB, insert as pg_insert
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column


class _Base(DeclarativeBase):
    pass


class AgentSessionTurn(_Base):
    __tablename__ = "agent_session_turns"

    source_id: Mapped[str] = mapped_column(sa.String, primary_key=True)
    session_id: Mapped[str] = mapped_column(sa.String, nullable=False, index=True)
    messages: Mapped[Any] = mapped_column(JSONB, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False
    )


class AgentSessionVar(_Base):
    __tablename__ = "agent_session_vars"

    session_id: Mapped[str] = mapped_column(sa.String, nullable=False)
    key: Mapped[str] = mapped_column(sa.String, nullable=False)
    value: Mapped[Any] = mapped_column(JSONB, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True), nullable=False
    )
    __table_args__ = (sa.PrimaryKeyConstraint("session_id", "key"),)


@contextmanager
def _get_session(dsn: str) -> Generator[Session, None, None]:
    engine = sa.create_engine(dsn)
    with Session(engine) as session:
        try:
            yield session
            session.commit()
        except Exception:
            session.rollback()
            raise


def _strip_thinking(history: list) -> list:
    """Remove ThinkingPart entries from ModelResponse messages in *history*.

    Thinking parts (chain-of-thought traces) are useful for the model that
    generated them but add tokens without benefit when replayed as context.
    """
    import dataclasses  # noqa: PLC0415
    from pydantic_ai.messages import ModelResponse, ThinkingPart  # noqa: PLC0415

    result = []
    for msg in history:
        if isinstance(msg, ModelResponse):
            clean = [p for p in msg.parts if not isinstance(p, ThinkingPart)]
            result.append(dataclasses.replace(msg, parts=clean))
        else:
            result.append(msg)
    return result


def load_history(session_id: str, dsn: str, *, strip_thinking: bool = True) -> list:
    """Load and concatenate all turns for *session_id* into a single message list.

    Returns an empty list if the session has no turns yet.
    The returned list is suitable for passing directly as ``message_history``
    to ``pydantic_ai.Agent.run()`` or ``run_sync()``.

    Args:
        strip_thinking: When ``True`` (default), ``ThinkingPart`` entries are
            removed from replayed ``ModelResponse`` messages.  Thinking traces
            are not useful as context and inflate the prompt token count.
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter  # noqa: PLC0415

    logger.info("Loading session history for session_id=%r", session_id)
    engine = sa.create_engine(dsn)
    with Session(engine) as db:
        rows = db.scalars(
            sa.select(AgentSessionTurn)
            .where(AgentSessionTurn.session_id == session_id)
            .order_by(AgentSessionTurn.created_at)
        ).all()

    history: list = []
    for row in rows:
        turn_msgs = ModelMessagesTypeAdapter.validate_python(row.messages)
        history.extend(turn_msgs)

    if strip_thinking:
        history = _strip_thinking(history)

    logger.info(
        "Session %r: loaded %d turn(s), %d message(s) total",
        session_id, len(rows), len(history),
    )
    return history


def append_turn(source_id: str, session_id: str, messages: list, dsn: str) -> None:
    """Persist *messages* (``result.new_messages()``) for one turn.

    No-op if *source_id* already exists — idempotent by construction.
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter  # noqa: PLC0415

    serialized = ModelMessagesTypeAdapter.dump_python(messages, mode="json")

    with _get_session(dsn) as db:
        stmt = (
            pg_insert(AgentSessionTurn)
            .values(
                source_id=source_id,
                session_id=session_id,
                messages=serialized,
                created_at=datetime.now(timezone.utc),
            )
            .on_conflict_do_nothing(index_elements=["source_id"])
        )
        db.execute(stmt)


def context_size(messages: list) -> tuple[float, int]:
    """Return ``(kb, tokens)`` for *messages*.

    *kb* is the UTF-8 byte size of the JSON-serialised history.
    *tokens* is a rough estimate (chars / 4 — the standard rule of thumb).
    Both are 0 when *messages* is empty.
    """
    from pydantic_ai.messages import ModelMessagesTypeAdapter  # noqa: PLC0415

    if not messages:
        return 0.0, 0
    serialized = ModelMessagesTypeAdapter.dump_python(messages, mode="json")
    text = json.dumps(serialized)
    kb = len(text.encode()) / 1024
    tokens = len(text) // 4
    return kb, tokens


def _truncate(text: str, limit: int = 88) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _message_debug_excerpt(message: Any) -> str:
    from pydantic_ai.messages import RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart, UserPromptPart  # noqa: PLC0415

    snippets: list[str] = []
    for part in getattr(message, "parts", []):
        if isinstance(part, (TextPart, UserPromptPart, RetryPromptPart)):
            content = getattr(part, "content", "")
            if isinstance(content, str) and content.strip():
                snippets.append(_truncate(content))
        elif isinstance(part, ToolCallPart):
            snippets.append(f"tool_call:{part.tool_name}")
        elif isinstance(part, ToolReturnPart):
            tool_name = getattr(part, "tool_name", "tool")
            snippets.append(f"tool_return:{tool_name}")
    return " | ".join(snippets[:3]) or "—"


def _analyse_history_messages(raw_messages: list, replay_messages: list) -> dict:
    """Return a troubleshooting summary for a session history."""
    from pydantic_ai.messages import (  # noqa: PLC0415
        ModelRequest,
        ModelResponse,
        RetryPromptPart,
        TextPart,
        ThinkingPart,
        ToolCallPart,
    )

    counters = Counter()
    issues: list[str] = []
    messages: list[dict[str, Any]] = []
    previous_kind: str | None = None

    for idx, (raw_msg, replay_msg) in enumerate(zip(raw_messages, replay_messages), start=1):
        raw_parts = list(getattr(raw_msg, "parts", []))
        replay_parts = list(getattr(replay_msg, "parts", []))
        raw_part_names = [type(part).__name__ for part in raw_parts]
        raw_kind = "request" if isinstance(raw_msg, ModelRequest) else "response"
        message_issues: list[str] = []

        if previous_kind == raw_kind:
            counters["consecutive_same_kind"] += 1
            message_issues.append("same kind as previous message")
        previous_kind = raw_kind

        counters[f"{raw_kind}_messages"] += 1
        counters["messages_total"] += 1

        if isinstance(raw_msg, ModelRequest):
            retry_parts = [part for part in raw_parts if isinstance(part, RetryPromptPart)]
            if retry_parts:
                counters["retry_prompt_requests"] += 1
                message_issues.append("contains retry prompt")
        else:
            text_parts = [part for part in raw_parts if isinstance(part, TextPart)]
            thinking_parts = [part for part in raw_parts if isinstance(part, ThinkingPart)]
            tool_call_parts = [part for part in raw_parts if isinstance(part, ToolCallPart)]
            text_content = "".join(part.content for part in text_parts)

            if not raw_parts:
                counters["empty_response_parts"] += 1
                message_issues.append("response has no parts")
            if text_parts:
                if not text_content.strip():
                    counters["blank_text_responses"] += 1
                    message_issues.append("response text is blank")
            else:
                counters["responses_without_text"] += 1
                message_issues.append("response has no text part")
            if thinking_parts and len(thinking_parts) == len(raw_parts):
                counters["thinking_only_responses"] += 1
                message_issues.append("thinking-only response")
            if tool_call_parts and not text_parts:
                counters["tool_call_only_responses"] += 1
                message_issues.append("tool-call-only response")
            if raw_parts and not replay_parts:
                counters["responses_empty_after_strip"] += 1
                message_issues.append("becomes empty after strip_thinking")

        messages.append(
            {
                "index": idx,
                "kind": raw_kind,
                "parts": raw_part_names,
                "excerpt": _message_debug_excerpt(raw_msg),
                "issues": message_issues,
            }
        )

    if raw_messages and isinstance(raw_messages[-1], ModelRequest):
        counters["dangling_request_at_end"] += 1
        issues.append("History ends with a request and no matching response.")
    if counters["retry_prompt_requests"]:
        issues.append(
            f"Found {counters['retry_prompt_requests']} request message(s) containing RetryPromptPart."
        )
    if counters["responses_without_text"]:
        issues.append(
            f"Found {counters['responses_without_text']} response message(s) without a TextPart."
        )
    if counters["blank_text_responses"]:
        issues.append(
            f"Found {counters['blank_text_responses']} response message(s) whose text is blank."
        )
    if counters["thinking_only_responses"]:
        issues.append(
            f"Found {counters['thinking_only_responses']} thinking-only response message(s)."
        )
    if counters["tool_call_only_responses"]:
        issues.append(
            f"Found {counters['tool_call_only_responses']} tool-call-only response message(s)."
        )
    if counters["responses_empty_after_strip"]:
        issues.append(
            f"Found {counters['responses_empty_after_strip']} response message(s) that become empty after strip_thinking."
        )
    if counters["consecutive_same_kind"]:
        issues.append(
            f"Found {counters['consecutive_same_kind']} place(s) where history has consecutive request/request or response/response messages."
        )

    return {
        "counts": dict(counters),
        "issues": issues,
        "messages": messages,
    }


def inspect_session_history(session_id: str, dsn: str, *, tail: int = 12) -> dict:
    """Return a structured troubleshooting report for *session_id*."""
    from pydantic_ai.messages import ModelMessagesTypeAdapter  # noqa: PLC0415

    logger.info("Inspecting session history for session_id=%r", session_id)
    engine = sa.create_engine(dsn)
    with Session(engine) as db:
        rows = db.scalars(
            sa.select(AgentSessionTurn)
            .where(AgentSessionTurn.session_id == session_id)
            .order_by(AgentSessionTurn.created_at)
        ).all()

    raw_messages: list = []
    turns: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        turn_messages = ModelMessagesTypeAdapter.validate_python(row.messages)
        raw_messages.extend(turn_messages)
        turns.append(
            {
                "index": idx,
                "source_id": row.source_id,
                "created_at": row.created_at.isoformat(),
                "message_count": len(turn_messages),
                "message_types": [type(msg).__name__ for msg in turn_messages],
            }
        )

    replay_messages = _strip_thinking(raw_messages)
    replay_payload = ModelMessagesTypeAdapter.dump_python(replay_messages, mode="json")
    raw_kb, raw_tokens = context_size(raw_messages)
    replay_kb, replay_tokens = context_size(replay_messages)
    analysis = _analyse_history_messages(raw_messages, replay_messages)

    return {
        "session_id": session_id,
        "turn_count": len(rows),
        "raw_message_count": len(raw_messages),
        "replay_message_count": len(replay_messages),
        "raw_context_kb": raw_kb,
        "raw_context_tokens": raw_tokens,
        "replay_context_kb": replay_kb,
        "replay_context_tokens": replay_tokens,
        "session_vars": load_session_vars(session_id, dsn),
        "issues": analysis["issues"],
        "counts": analysis["counts"],
        "recent_messages": analysis["messages"][-tail:] if tail > 0 else [],
        "replay_payload": replay_payload,
        "turns": turns,
    }


def delete_session(session_id: str, dsn: str) -> int:
    """Delete all turns for *session_id*.

    Returns the number of rows deleted (0 if the session did not exist).
    """
    with _get_session(dsn) as db:
        result = db.execute(
            sa.delete(AgentSessionTurn).where(
                AgentSessionTurn.session_id == session_id
            )
        )
    return result.rowcount


# ── Session variable persistence ─────────────────────────────────────────────


def load_session_vars(session_id: str, dsn: str) -> dict:
    """Load all session variables for *session_id*.

    Returns a plain ``dict`` mapping key to the stored value (bool, int,
    float, or str as originally set).  Returns an empty dict when no variables
    have been saved yet.
    """
    engine = sa.create_engine(dsn)
    with Session(engine) as db:
        rows = db.scalars(
            sa.select(AgentSessionVar).where(
                AgentSessionVar.session_id == session_id
            )
        ).all()
    return {row.key: row.value for row in rows}


def save_session_var(session_id: str, key: str, value: Any, dsn: str) -> None:
    """Upsert a single session variable.

    Inserts a new row, or updates ``value`` and ``updated_at`` if the
    ``(session_id, key)`` pair already exists.
    """
    with _get_session(dsn) as db:
        stmt = (
            pg_insert(AgentSessionVar)
            .values(
                session_id=session_id,
                key=key,
                value=value,
                updated_at=datetime.now(timezone.utc),
            )
            .on_conflict_do_update(
                index_elements=["session_id", "key"],
                set_={
                    "value": value,
                    "updated_at": datetime.now(timezone.utc),
                },
            )
        )
        db.execute(stmt)


def delete_session_var(session_id: str, key: str, dsn: str) -> int:
    """Delete a single session variable.

    Returns 1 if the variable existed and was deleted, 0 otherwise.
    """
    with _get_session(dsn) as db:
        result = db.execute(
            sa.delete(AgentSessionVar).where(
                sa.and_(
                    AgentSessionVar.session_id == session_id,
                    AgentSessionVar.key == key,
                )
            )
        )
    return result.rowcount
