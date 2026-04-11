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
