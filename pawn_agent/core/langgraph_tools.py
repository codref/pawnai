"""LangGraph-specific tool adapters and node builders."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Awaitable
from typing import Any, Callable

from pawn_agent.core.burr_chat import _normalize_output
from pawn_agent.core.langgraph_state import (
    ensure_langgraph_state,
    get_state_field,
    serialize_langgraph_state,
    set_state_fields,
)
from pawn_agent.tools.analyze_summary import analyze_summary_impl
from pawn_agent.tools.list_sessions import list_sessions_impl
from pawn_agent.tools.query_conversation import query_conversation_impl
from pawn_agent.tools.save_to_siyuan import save_to_siyuan_impl
from pawn_agent.utils.config import AgentConfig


def _resolve_session_id_from_state(state: Mapping[str, object]) -> str:
    """Resolve the session id from existing LangGraph state only."""
    latest_user_message = _normalize_output(get_state_field(state, "latest_user_message")).strip()
    requested_session_id = _normalize_output(get_state_field(state, "requested_session_id")).strip()
    latest_session_id = _normalize_output(get_state_field(state, "latest_session_id")).strip()
    session_catalog_output = _normalize_output(get_state_field(state, "session_catalog_output"))

    if _looks_like_explicit_session_id(requested_session_id):
        return requested_session_id
    if _is_confirmation_prompt(latest_user_message) and latest_session_id:
        return latest_session_id
    if session_catalog_output:
        resolved_session_id = resolve_session_id_from_list_output(
            latest_user_message or requested_session_id,
            session_catalog_output,
        )
        if resolved_session_id:
            return resolved_session_id
    if latest_session_id:
        return latest_session_id
    return ""


def _looks_like_explicit_session_id(value: str) -> bool:
    candidate = _normalize_output(value).strip()
    if not candidate:
        return False
    return any(ch.isdigit() for ch in candidate) and any(sep in candidate for sep in "-_:")


def _is_confirmation_prompt(user_prompt: str) -> bool:
    normalized = _normalize_output(user_prompt).strip().lower()
    return normalized in {
        "yes",
        "y",
        "yeah",
        "yep",
        "sure",
        "ok",
        "okay",
        "please do",
        "do it",
        "go ahead",
    }


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


def _resolve_session_id_from_catalog_output(user_prompt: str, tool_output: str) -> str:
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


def resolve_session_id_from_list_output(user_prompt: str, tool_output: str) -> str:
    """Public compatibility wrapper for resolving a session id from catalog output."""
    return _resolve_session_id_from_catalog_output(user_prompt, tool_output)


def run_list_sessions_tool(cfg: AgentConfig) -> str:
    """Execute the list-sessions tool for LangGraph mode."""
    return list_sessions_impl(cfg)


def run_query_conversation_tool(cfg: AgentConfig, session_id: str) -> str:
    """Execute the query-conversation tool for LangGraph mode."""
    return query_conversation_impl(cfg, session_id)


async def run_analyze_summary_tool(
    cfg: AgentConfig,
    session_id: str,
    *,
    save: bool = False,
    title: str | None = None,
) -> str:
    """Execute the standard structured-analysis tool for LangGraph mode."""
    return await analyze_summary_impl(cfg, session_id, save=save, title=title)


def run_save_to_siyuan_tool(
    cfg: AgentConfig,
    session_id: str,
    content: str,
    title: str | None = None,
) -> str:
    """Execute the save-to-SiYuan tool for LangGraph mode."""
    return save_to_siyuan_impl(cfg, session_id=session_id, content=content, title=title, path=None)


def _bootstrap_session_catalog_if_needed(
    state: Mapping[str, object],
    cfg: AgentConfig,
) -> dict[str, Any]:
    """Populate session catalog output when it is missing."""
    current = ensure_langgraph_state(state)
    session_catalog_output = _normalize_output(get_state_field(current, "session_catalog_output"))
    if not session_catalog_output.strip():
        session_catalog_output = run_list_sessions_tool(cfg)
        current = set_state_fields(
            dict(current),
            session_catalog_output=session_catalog_output,
        )
    return current


def resolve_session_id(
    state: Mapping[str, object],
    cfg: AgentConfig,
    *,
    bootstrap_catalog: bool = True,
) -> tuple[dict[str, Any], str]:
    """Resolve a session id for a LangGraph tool call.

    Returns the normalized state and the resolved session id. When requested,
    the resolver can bootstrap the session catalog during the same turn.
    """
    current = ensure_langgraph_state(state)
    session_id = _resolve_session_id_from_state(current)
    if session_id or not bootstrap_catalog:
        return current, session_id

    current = _bootstrap_session_catalog_if_needed(current, cfg)
    latest_user_message = _normalize_output(get_state_field(current, "latest_user_message"))
    session_id = _resolve_session_id_from_catalog_output(
        latest_user_message,
        _normalize_output(get_state_field(current, "session_catalog_output")),
    )
    return current, session_id


def build_tool_list_sessions_node(
    *,
    cfg: AgentConfig,
    tracer=None,
    trace_full_state: bool = False,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the LangGraph node for listing sessions."""

    def tool_list_sessions_node(state: dict[str, Any]) -> dict[str, Any]:
        current = ensure_langgraph_state(state)
        latest_user_message = _normalize_output(get_state_field(current, "latest_user_message"))
        if tracer is None:
            tool_output = run_list_sessions_tool(cfg)
            updated = set_state_fields(
                dict(current),
                tool_name="list_sessions",
                tool_output=tool_output,
                session_catalog_output=tool_output,
            )
            resolved_session_id = _resolve_session_id_from_catalog_output(
                latest_user_message,
                tool_output,
            )
            if resolved_session_id:
                updated = set_state_fields(updated, latest_session_id=resolved_session_id)
            return updated
        with tracer.start_as_current_span("langgraph-tool-list-sessions") as span:
            if trace_full_state:
                span.set_attribute("state.before.json", serialize_langgraph_state(current))
            tool_output = run_list_sessions_tool(cfg)
            updated = set_state_fields(
                dict(current),
                tool_name="list_sessions",
                tool_output=tool_output,
                session_catalog_output=tool_output,
            )
            resolved_session_id = _resolve_session_id_from_catalog_output(
                latest_user_message,
                tool_output,
            )
            if resolved_session_id:
                updated = set_state_fields(updated, latest_session_id=resolved_session_id)
            span.set_attribute("tool.name", "list_sessions")
            if resolved_session_id:
                span.set_attribute("session.id", resolved_session_id)
            span.set_attribute("output.value", tool_output)
            if trace_full_state:
                span.set_attribute("state.after.json", serialize_langgraph_state(updated))
            return updated

    return tool_list_sessions_node


def build_tool_query_conversation_node(
    *,
    cfg: AgentConfig,
    tracer=None,
    trace_full_state: bool = False,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the LangGraph node for loading a conversation transcript."""

    def tool_query_conversation_node(state: dict[str, Any]) -> dict[str, Any]:
        current = ensure_langgraph_state(state)
        current, session_id = resolve_session_id(current, cfg)
        if tracer is None:
            if not session_id:
                return set_state_fields(
                    dict(current),
                    tool_name="query_conversation",
                    tool_output=(
                    "I need a session id to retrieve a conversation. "
                    "Ask for available sessions first or specify the session id."
                    ),
                )
            tool_output = run_query_conversation_tool(cfg, session_id)
            return set_state_fields(
                dict(current),
                tool_name="query_conversation",
                tool_output=tool_output,
                latest_session_id=session_id,
            )
        with tracer.start_as_current_span("langgraph-tool-query-conversation") as span:
            if trace_full_state:
                span.set_attribute("state.before.json", serialize_langgraph_state(current))
            current, session_id = resolve_session_id(current, cfg)
            if not session_id:
                updated = set_state_fields(
                    dict(current),
                    tool_name="query_conversation",
                    tool_output=(
                    "I need a session id to retrieve a conversation. "
                    "Ask for available sessions first or specify the session id."
                    ),
                )
                span.set_attribute("tool.name", "query_conversation")
                span.set_attribute("output.value", _normalize_output(get_state_field(updated, "tool_output")))
                if trace_full_state:
                    span.set_attribute("state.after.json", serialize_langgraph_state(updated))
                return updated
            tool_output = run_query_conversation_tool(cfg, session_id)
            updated = set_state_fields(
                dict(current),
                tool_name="query_conversation",
                tool_output=tool_output,
                latest_session_id=session_id,
            )
            span.set_attribute("tool.name", "query_conversation")
            span.set_attribute("session.id", session_id)
            span.set_attribute("output.value", tool_output)
            if trace_full_state:
                span.set_attribute("state.after.json", serialize_langgraph_state(updated))
            return updated

    return tool_query_conversation_node


def build_tool_analyze_summary_node(
    *,
    cfg: AgentConfig,
    tracer=None,
    trace_full_state: bool = False,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Build the LangGraph node for the standard structured session analysis."""

    async def tool_analyze_summary_node(state: dict[str, Any]) -> dict[str, Any]:
        current = ensure_langgraph_state(state)
        current, session_id = resolve_session_id(current, cfg)
        if tracer is None:
            if not session_id:
                return set_state_fields(
                    dict(current),
                    tool_name="analyze_summary",
                    tool_output=(
                        "I need a session id to run the standard analysis. "
                        "Ask for available sessions first or specify the session id."
                    ),
                )
            tool_output = await run_analyze_summary_tool(cfg, session_id)
            return set_state_fields(
                dict(current),
                tool_name="analyze_summary",
                tool_output=tool_output,
                latest_session_id=session_id,
            )
        with tracer.start_as_current_span("langgraph-tool-analyze-summary") as span:
            if trace_full_state:
                span.set_attribute("state.before.json", serialize_langgraph_state(current))
            if not session_id:
                updated = set_state_fields(
                    dict(current),
                    tool_name="analyze_summary",
                    tool_output=(
                        "I need a session id to run the standard analysis. "
                        "Ask for available sessions first or specify the session id."
                    ),
                )
                span.set_attribute("tool.name", "analyze_summary")
                span.set_attribute(
                    "output.value",
                    _normalize_output(get_state_field(updated, "tool_output")),
                )
                if trace_full_state:
                    span.set_attribute("state.after.json", serialize_langgraph_state(updated))
                return updated
            tool_output = await run_analyze_summary_tool(cfg, session_id)
            updated = set_state_fields(
                dict(current),
                tool_name="analyze_summary",
                tool_output=tool_output,
                latest_session_id=session_id,
            )
            span.set_attribute("tool.name", "analyze_summary")
            span.set_attribute("session.id", session_id)
            span.set_attribute("output.value", tool_output)
            if trace_full_state:
                span.set_attribute("state.after.json", serialize_langgraph_state(updated))
            return updated

    return tool_analyze_summary_node


def build_tool_save_to_siyuan_node(
    *,
    cfg: AgentConfig,
    tracer=None,
    trace_full_state: bool = False,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the LangGraph node for saving generated content to SiYuan."""

    def tool_save_to_siyuan_node(state: dict[str, Any]) -> dict[str, Any]:
        current = ensure_langgraph_state(state)
        session_id = _normalize_output(get_state_field(current, "latest_session_id")).strip()
        content = _normalize_output(get_state_field(current, "latest_generated_content"))
        title = _normalize_output(get_state_field(current, "latest_generated_title")).strip() or None
        if tracer is None:
            if not session_id:
                return set_state_fields(
                    dict(current),
                    tool_name="save_to_siyuan",
                    pending_save_to_siyuan=False,
                    tool_output=(
                    "I need a session in focus before I can save to SiYuan. "
                    "Retrieve or analyze a session first."
                    ),
                )
            if not content.strip():
                return set_state_fields(
                    dict(current),
                    tool_name="save_to_siyuan",
                    pending_save_to_siyuan=False,
                    tool_output=(
                    "I need generated content to save to SiYuan. "
                    "Ask me to produce the report or analysis first."
                    ),
                )
            tool_output = run_save_to_siyuan_tool(cfg, session_id, content, title=title)
            return set_state_fields(
                dict(current),
                tool_name="save_to_siyuan",
                pending_save_to_siyuan=False,
                tool_output=tool_output,
            )
        with tracer.start_as_current_span("langgraph-tool-save-to-siyuan") as span:
            if trace_full_state:
                span.set_attribute("state.before.json", serialize_langgraph_state(current))
            if not session_id:
                updated = set_state_fields(
                    dict(current),
                    tool_name="save_to_siyuan",
                    pending_save_to_siyuan=False,
                    tool_output=(
                    "I need a session in focus before I can save to SiYuan. "
                    "Retrieve or analyze a session first."
                    ),
                )
                span.set_attribute("tool.name", "save_to_siyuan")
                span.set_attribute("output.value", _normalize_output(get_state_field(updated, "tool_output")))
                if trace_full_state:
                    span.set_attribute("state.after.json", serialize_langgraph_state(updated))
                return updated
            if not content.strip():
                updated = set_state_fields(
                    dict(current),
                    tool_name="save_to_siyuan",
                    pending_save_to_siyuan=False,
                    tool_output=(
                    "I need generated content to save to SiYuan. "
                    "Ask me to produce the report or analysis first."
                    ),
                )
                span.set_attribute("tool.name", "save_to_siyuan")
                span.set_attribute("session.id", session_id)
                span.set_attribute("output.value", _normalize_output(get_state_field(updated, "tool_output")))
                if trace_full_state:
                    span.set_attribute("state.after.json", serialize_langgraph_state(updated))
                return updated
            tool_output = run_save_to_siyuan_tool(cfg, session_id, content, title=title)
            updated = set_state_fields(
                dict(current),
                tool_name="save_to_siyuan",
                pending_save_to_siyuan=False,
                tool_output=tool_output,
            )
            span.set_attribute("tool.name", "save_to_siyuan")
            span.set_attribute("session.id", session_id)
            span.set_attribute("output.value", tool_output)
            if trace_full_state:
                span.set_attribute("state.after.json", serialize_langgraph_state(updated))
            return updated

    return tool_save_to_siyuan_node
