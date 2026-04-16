"""In-memory session variable store with optional DB persistence.

Variables are key-value pairs scoped to a single chat session. They persist
across restarts when a ``session_id`` and ``dsn`` are provided; otherwise
they are ephemeral for the lifetime of the process.

Typical use::

    vars = SessionVars(session_id="my-session", dsn=cfg.db_dsn)
    vars.load()                      # hydrate from DB
    vars.set("listen_only", "true")  # coerced to bool, persisted
    vars.build_instructions()        # per-turn system-prompt fragment
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

_KEY_RE = re.compile(r"^[a-zA-Z_][a-zA-Z0-9_]*$")


def _coerce(value: str) -> bool | int | float | str:
    """Coerce a raw string to the most specific scalar type.

    Order matters: booleans must be checked before int so that "1"/"0"
    map to ``True``/``False`` rather than the integers 1/0.
    """
    low = value.lower()
    if low in {"true", "yes", "1"}:
        return True
    if low in {"false", "no", "0"}:
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


class SessionVars:
    """Ephemeral key-value store for per-session agent configuration.

    Args:
        session_id: When set, changes are persisted to the DB and the store
            can be hydrated via :meth:`load`.  ``None`` gives in-memory-only
            behaviour (suitable for anonymous ``run`` invocations).
        dsn: PostgreSQL DSN.  Required for persistence; ignored when
            ``session_id`` is ``None``.
    """

    def __init__(
        self,
        session_id: str | None = None,
        dsn: str | None = None,
    ) -> None:
        self._store: dict[str, bool | int | float | str] = {}
        self._session_id = session_id
        self._dsn = dsn

    # ── Persistence ──────────────────────────────────────────────────────────

    def load(self) -> None:
        """Hydrate the in-memory store from the DB.

        No-op when ``session_id`` or ``dsn`` is unset.
        """
        if not (self._session_id and self._dsn):
            return
        from pawn_agent.core.session_store import load_session_vars  # noqa: PLC0415

        data = load_session_vars(self._session_id, self._dsn)
        self._store.update(data)
        if data:
            logger.info(
                "Loaded %d session var(s) for session %r",
                len(data),
                self._session_id,
            )

    def _persist_set(self, key: str, value: Any) -> None:
        if not (self._session_id and self._dsn):
            return
        from pawn_agent.core.session_store import save_session_var  # noqa: PLC0415

        save_session_var(self._session_id, key, value, self._dsn)

    def _persist_delete(self, key: str) -> None:
        if not (self._session_id and self._dsn):
            return
        from pawn_agent.core.session_store import delete_session_var  # noqa: PLC0415

        delete_session_var(self._session_id, key, self._dsn)

    # ── Mutations ─────────────────────────────────────────────────────────────

    def set(self, key: str, value: str) -> str:
        """Set *key* to *value*, coercing to the appropriate scalar type.

        Returns a human-readable confirmation or an error string on invalid key.
        """
        if not _KEY_RE.match(key):
            return (
                f"Invalid key {key!r}: must match [a-zA-Z_][a-zA-Z0-9_]*"
            )
        coerced = _coerce(value)
        self._store[key] = coerced
        self._persist_set(key, coerced)
        return f"Set {key} = {coerced!r}  ({type(coerced).__name__})"

    def unset(self, key: str) -> str:
        """Remove *key* from the store.

        Returns a confirmation, or a message if the key was not set.
        """
        if key not in self._store:
            return f"{key!r} is not set"
        del self._store[key]
        self._persist_delete(key)
        return f"Unset {key}"

    # ── Reads ─────────────────────────────────────────────────────────────────

    def get(self, key: str, default: Any = None) -> Any:
        return self._store.get(key, default)

    def get_bool(self, key: str, default: bool = False) -> bool:
        val = self._store.get(key, default)
        if isinstance(val, bool):
            return val
        if isinstance(val, str):
            return val.lower() in {"true", "yes", "1"}
        return bool(val)

    def list_vars(self) -> dict:
        """Return a shallow copy of the current store."""
        return dict(self._store)

    # ── Formatting ────────────────────────────────────────────────────────────

    def format_for_display(self) -> str:
        """Human-readable table for the ``/vars`` REPL command."""
        if not self._store:
            return "No session variables set."
        lines = [
            f"  {k} = {v!r}  ({type(v).__name__})"
            for k, v in sorted(self._store.items())
        ]
        return "\n".join(lines)

    def format_for_llm(self) -> str:
        """Compact JSON representation for tool return values."""
        return json.dumps(self._store, default=str)

    def build_instructions(self) -> str | None:
        """Build a per-turn system-prompt fragment from the active variables.

        Returns ``None`` when no variables are set (so callers can skip the
        ``instructions`` parameter entirely and avoid an empty string being
        injected).

        The fragment always lists all active vars as JSON context, then
        appends directive text for known behavioral flags:

        - ``listen_only=true`` — instructs the LLM to observe silently and
          only respond when directly addressed.
        - ``verbosity=quiet`` — requests concise responses.
        - ``verbosity=debug`` — requests detailed, explicit reasoning.
        """
        if not self._store:
            return None

        parts: list[str] = [f"Active session variables: {self.format_for_llm()}"]

        if self.get_bool("listen_only"):
            parts.append(
                "You are in listen-only mode (observing a think-aloud session). "
                "This message was routed to you because it appears to be directed at you."
            )

        verbosity = self._store.get("verbosity", "")
        if isinstance(verbosity, str):
            if verbosity.lower() == "quiet":
                parts.append("Keep your responses concise and avoid elaboration.")
            elif verbosity.lower() == "debug":
                parts.append(
                    "Be detailed and explicit about your reasoning and intermediate steps."
                )

        return "\n\n".join(parts)
