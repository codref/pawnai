"""Agent tool for reading and modifying session variables.

Session variables are key-value pairs scoped to the current chat session.
They influence agent behaviour (e.g. ``listen_only``, ``verbosity``) and
persist across restarts of the same named session.
"""

from __future__ import annotations

from pydantic_ai import Tool

from pawn_agent.utils.config import AgentConfig

NAME = "session_vars"
DESCRIPTION = "Read or modify session variables for this conversation."


def build(cfg: AgentConfig, *, session_vars) -> Tool:  # noqa: ARG001
    """Return the session_vars tool, closing over the live *session_vars* store."""

    def session_vars_tool(
        action: str,
        key: str = "",
        value: str = "",
    ) -> str:
        """Read or write session variables for this conversation.

        Session variables are key-value pairs that influence agent behaviour
        and persist across restarts of the same named session.

        Actions:
        - "get":   Return the current value of *key*.
        - "set":   Set *key* to *value* (auto-coerced to bool/int/float/str).
        - "unset": Remove *key* from the session.
        - "list":  Return all current session variables as JSON.

        Common variables:
        - listen_only (bool)  — when true, observe silently and only respond
                                if directly addressed.
        - verbosity   (str)   — "quiet" for concise replies, "debug" for
                                detailed reasoning, omit for default behaviour.

        Args:
            action: One of "get", "set", "unset", "list".
            key: Variable name (required for get/set/unset; ignored for list).
            value: New value string (required for set; ignored otherwise).
        """
        if action == "list":
            return session_vars.format_for_llm()

        if action == "get":
            if not key:
                return "Error: key is required for action 'get'"
            val = session_vars.get(key)
            if val is None:
                return f"{key!r} is not set"
            return f"{key} = {val!r}  ({type(val).__name__})"

        if action == "set":
            if not key:
                return "Error: key is required for action 'set'"
            return session_vars.set(key, value)

        if action == "unset":
            if not key:
                return "Error: key is required for action 'unset'"
            return session_vars.unset(key)

        return (
            f"Unknown action {action!r}. Must be one of: get, set, unset, list."
        )

    return Tool(session_vars_tool)
