"""LangGraph-specific tool adapters and node builders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Callable

from pawn_agent.core.burr_chat import _normalize_output
from pawn_agent.tools.list_sessions import list_sessions_impl
from pawn_agent.tools.query_conversation import query_conversation_impl
from pawn_agent.tools.save_to_siyuan import save_to_siyuan_impl
from pawn_agent.utils.config import AgentConfig


def resolve_session_id_for_tool(state: Mapping[str, object]) -> str:
    """Resolve the session id for a session-scoped LangGraph tool call."""
    requested_session_id = _normalize_output(state.get("requested_session_id", "")).strip()
    if requested_session_id:
        return requested_session_id
    return _normalize_output(state.get("latest_session_id", "")).strip()


def _session_ids_from_list_output(tool_output: str) -> list[str]:
    session_ids: list[str] = []
    for raw_line in _normalize_output(tool_output).splitlines():
        line = raw_line.strip()
        if not line or line.lower().startswith("found "):
            continue
        session_id = line.split("|", 1)[0].strip()
        if session_id:
            session_ids.append(session_id)
    return session_ids


def resolve_session_id_from_list_output(user_prompt: str, tool_output: str) -> str:
    """Resolve a likely target session id from list-sessions output."""
    session_ids = _session_ids_from_list_output(tool_output)
    if not session_ids:
        return ""
    prompt_tokens = [
        token
        for token in "".join(
            ch.lower() if ch.isalnum() or ch in {"-", "_", ":"} else " "
            for ch in _normalize_output(user_prompt)
        ).split()
        if len(token) >= 3
        and token
        not in {
            "the",
            "and",
            "with",
            "from",
            "into",
            "this",
            "that",
            "show",
            "latest",
            "last",
            "most",
            "recent",
            "session",
            "sessions",
            "retrieve",
            "get",
            "give",
            "conversation",
        }
    ]
    for session_id in session_ids:
        lowered = session_id.lower()
        if any(token in lowered for token in prompt_tokens):
            return session_id
    return session_ids[0]


def run_list_sessions_tool(cfg: AgentConfig) -> str:
    """Execute the list-sessions tool for LangGraph mode."""
    return list_sessions_impl(cfg)


def run_query_conversation_tool(cfg: AgentConfig, session_id: str) -> str:
    """Execute the query-conversation tool for LangGraph mode."""
    return query_conversation_impl(cfg, session_id)


def run_save_to_siyuan_tool(
    cfg: AgentConfig,
    session_id: str,
    content: str,
    title: str | None = None,
) -> str:
    """Execute the save-to-SiYuan tool for LangGraph mode."""
    return save_to_siyuan_impl(cfg, session_id=session_id, content=content, title=title, path=None)


def build_tool_list_sessions_node(
    *,
    cfg: AgentConfig,
    tracer=None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the LangGraph node for listing sessions."""

    def tool_list_sessions_node(state: dict[str, Any]) -> dict[str, Any]:
        latest_user_message = _normalize_output(state.get("latest_user_message", ""))
        if tracer is None:
            tool_output = run_list_sessions_tool(cfg)
            updated = dict(state)
            updated["tool_name"] = "list_sessions"
            updated["tool_output"] = tool_output
            resolved_session_id = resolve_session_id_from_list_output(
                latest_user_message,
                tool_output,
            )
            if resolved_session_id:
                updated["latest_session_id"] = resolved_session_id
            return updated
        with tracer.start_as_current_span("langgraph-tool-list-sessions") as span:
            tool_output = run_list_sessions_tool(cfg)
            updated = dict(state)
            updated["tool_name"] = "list_sessions"
            updated["tool_output"] = tool_output
            resolved_session_id = resolve_session_id_from_list_output(
                latest_user_message,
                tool_output,
            )
            if resolved_session_id:
                updated["latest_session_id"] = resolved_session_id
            span.set_attribute("tool.name", "list_sessions")
            if resolved_session_id:
                span.set_attribute("session.id", resolved_session_id)
            span.set_attribute("output.value", tool_output)
            return updated

    return tool_list_sessions_node


def build_tool_query_conversation_node(
    *,
    cfg: AgentConfig,
    tracer=None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the LangGraph node for loading a conversation transcript."""

    def tool_query_conversation_node(state: dict[str, Any]) -> dict[str, Any]:
        session_id = resolve_session_id_for_tool(state)
        if tracer is None:
            updated = dict(state)
            if not session_id:
                updated["tool_name"] = "query_conversation"
                updated["tool_output"] = (
                    "I need a session id to retrieve a conversation. "
                    "Ask for available sessions first or specify the session id."
                )
                return updated
            tool_output = run_query_conversation_tool(cfg, session_id)
            updated["tool_name"] = "query_conversation"
            updated["tool_output"] = tool_output
            updated["latest_session_id"] = session_id
            return updated
        with tracer.start_as_current_span("langgraph-tool-query-conversation") as span:
            updated = dict(state)
            if not session_id:
                updated["tool_name"] = "query_conversation"
                updated["tool_output"] = (
                    "I need a session id to retrieve a conversation. "
                    "Ask for available sessions first or specify the session id."
                )
                span.set_attribute("tool.name", "query_conversation")
                span.set_attribute("output.value", updated["tool_output"])
                return updated
            tool_output = run_query_conversation_tool(cfg, session_id)
            updated["tool_name"] = "query_conversation"
            updated["tool_output"] = tool_output
            updated["latest_session_id"] = session_id
            span.set_attribute("tool.name", "query_conversation")
            span.set_attribute("session.id", session_id)
            span.set_attribute("output.value", tool_output)
            return updated

    return tool_query_conversation_node


def build_tool_save_to_siyuan_node(
    *,
    cfg: AgentConfig,
    tracer=None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the LangGraph node for saving generated content to SiYuan."""

    def tool_save_to_siyuan_node(state: dict[str, Any]) -> dict[str, Any]:
        session_id = _normalize_output(state.get("latest_session_id", "")).strip()
        content = _normalize_output(state.get("latest_generated_content", ""))
        title = _normalize_output(state.get("latest_generated_title", "")).strip() or None
        if tracer is None:
            updated = dict(state)
            if not session_id:
                updated["tool_name"] = "save_to_siyuan"
                updated["pending_save_to_siyuan"] = False
                updated["tool_output"] = (
                    "I need a session in focus before I can save to SiYuan. "
                    "Retrieve or analyze a session first."
                )
                return updated
            if not content.strip():
                updated["tool_name"] = "save_to_siyuan"
                updated["pending_save_to_siyuan"] = False
                updated["tool_output"] = (
                    "I need generated content to save to SiYuan. "
                    "Ask me to produce the report or analysis first."
                )
                return updated
            tool_output = run_save_to_siyuan_tool(cfg, session_id, content, title=title)
            updated["tool_name"] = "save_to_siyuan"
            updated["pending_save_to_siyuan"] = False
            updated["tool_output"] = tool_output
            return updated
        with tracer.start_as_current_span("langgraph-tool-save-to-siyuan") as span:
            updated = dict(state)
            if not session_id:
                updated["tool_name"] = "save_to_siyuan"
                updated["pending_save_to_siyuan"] = False
                updated["tool_output"] = (
                    "I need a session in focus before I can save to SiYuan. "
                    "Retrieve or analyze a session first."
                )
                span.set_attribute("tool.name", "save_to_siyuan")
                span.set_attribute("output.value", updated["tool_output"])
                return updated
            if not content.strip():
                updated["tool_name"] = "save_to_siyuan"
                updated["pending_save_to_siyuan"] = False
                updated["tool_output"] = (
                    "I need generated content to save to SiYuan. "
                    "Ask me to produce the report or analysis first."
                )
                span.set_attribute("tool.name", "save_to_siyuan")
                span.set_attribute("session.id", session_id)
                span.set_attribute("output.value", updated["tool_output"])
                return updated
            tool_output = run_save_to_siyuan_tool(cfg, session_id, content, title=title)
            updated["tool_name"] = "save_to_siyuan"
            updated["pending_save_to_siyuan"] = False
            updated["tool_output"] = tool_output
            span.set_attribute("tool.name", "save_to_siyuan")
            span.set_attribute("session.id", session_id)
            span.set_attribute("output.value", tool_output)
            return updated

    return tool_save_to_siyuan_node
