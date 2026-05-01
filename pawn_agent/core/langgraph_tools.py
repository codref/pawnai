"""LangGraph-specific tool adapters and node builders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re
from typing import Any, Awaitable, Callable, Literal

from pawn_agent.core.chat_primitives import normalize_output
from pawn_agent.core.langgraph_state import (
    ensure_langgraph_state,
    get_recent_messages,
    get_state_field,
    serialize_langgraph_state,
    set_state_fields,
)
from pawn_agent.core.session_candidates import (
    SessionCandidate,
    candidates_from_state,
    candidates_to_state,
    parse_session_candidates_from_list_output,
    render_session_catalog,
    session_search_text,
)
from pawn_agent.tools.analyze_summary import analyze_summary_impl
from pawn_agent.tools.list_sessions import list_session_candidates_impl, list_sessions_impl
from pawn_agent.tools.push_queue_message import push_queue_message_impl
from pawn_agent.tools.propose_schedule import propose_schedule_change_impl
from pawn_agent.tools.query_conversation import query_conversation_impl
from pawn_agent.tools.save_to_siyuan import save_to_siyuan_impl
from pawn_agent.tools.search_knowledge import search_knowledge_impl
from pawn_agent.tools.vectorize import vectorize_impl
from pawn_agent.utils.config import AgentConfig

_INTERVAL_RE = re.compile(
    r"\b(?:every|each)\s+(\d+)\s*(minutes?|mins?|minute|hours?|hrs?|hour)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SessionResolution:
    session_id: str | None
    confidence: float
    source: Literal[
        "explicit",
        "confirmation",
        "active",
        "catalog_match",
        "single_candidate",
        "none",
    ]
    needs_confirmation: bool = False


_SESSION_MATCH_STOPWORDS = {
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


def _session_lookup_tokens(text: str) -> list[str]:
    normalized = "".join(
        ch.lower() if ch.isalnum() or ch in {"-", "_", ":"} else " "
        for ch in normalize_output(text)
    )
    return [
        token
        for token in normalized.split()
        if len(token) >= 3 and token not in _SESSION_MATCH_STOPWORDS
    ]


def _has_recency_hint(text: str) -> bool:
    tokens = {
        token
        for token in "".join(
            ch.lower() if ch.isalnum() else " " for ch in normalize_output(text)
        ).split()
    }
    return bool(tokens & {"latest", "last", "most", "recent"})


def _is_confirmation_prompt(user_prompt: str) -> bool:
    normalized = normalize_output(user_prompt).strip().lower()
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


def _session_candidates_from_state(state: Mapping[str, object]) -> list[SessionCandidate]:
    candidates = candidates_from_state(get_state_field(state, "session_candidates"))
    if candidates:
        return candidates
    return parse_session_candidates_from_list_output(
        normalize_output(get_state_field(state, "session_catalog_output"))
    )


def _session_ids_from_candidates(candidates: list[SessionCandidate]) -> list[str]:
    return [candidate.session_id for candidate in candidates]


def _session_ids_from_list_output(tool_output: str) -> list[str]:
    return _session_ids_from_candidates(parse_session_candidates_from_list_output(tool_output))


def _exact_session_id_match(
    value: str,
    candidates: list[SessionCandidate],
) -> str | None:
    requested = normalize_output(value).strip()
    if not requested:
        return None
    known_ids = {candidate.session_id for candidate in candidates}
    return requested if requested in known_ids else None


def _resolve_session_from_candidates(
    user_prompt: str,
    candidates: list[SessionCandidate],
) -> SessionResolution:
    if not candidates:
        return SessionResolution(None, 0.0, "none")

    prompt_tokens = _session_lookup_tokens(user_prompt)
    matches: list[SessionCandidate] = []
    for candidate in candidates:
        search_text = session_search_text(candidate).lower()
        if any(token in search_text for token in prompt_tokens):
            matches.append(candidate)

    if len(matches) == 1:
        return SessionResolution(matches[0].session_id, 0.8, "catalog_match")
    if len(matches) > 1 and _has_recency_hint(user_prompt):
        return SessionResolution(matches[0].session_id, 0.75, "catalog_match")
    if _has_recency_hint(user_prompt):
        return SessionResolution(candidates[0].session_id, 0.75, "catalog_match")
    if len(candidates) == 1:
        return SessionResolution(candidates[0].session_id, 0.6, "single_candidate")
    return SessionResolution(None, 0.0, "none", needs_confirmation=bool(matches))


def _resolve_session_from_state(state: Mapping[str, object]) -> SessionResolution:
    """Resolve the session id from existing LangGraph state only."""
    latest_user_message = normalize_output(get_state_field(state, "latest_user_message")).strip()
    requested_session_id = normalize_output(get_state_field(state, "requested_session_id")).strip()
    latest_session_id = normalize_output(get_state_field(state, "latest_session_id")).strip()
    candidates = _session_candidates_from_state(state)

    exact_match = _exact_session_id_match(requested_session_id, candidates)
    if exact_match:
        return SessionResolution(exact_match, 1.0, "explicit")
    if requested_session_id and not candidates:
        return SessionResolution(requested_session_id, 0.7, "explicit")
    if _is_confirmation_prompt(latest_user_message) and latest_session_id:
        return SessionResolution(latest_session_id, 0.95, "confirmation")
    if candidates and (
        requested_session_id
        or _has_recency_hint(latest_user_message)
        or _session_lookup_tokens(latest_user_message)
    ):
        catalog_resolution = _resolve_session_from_candidates(
            latest_user_message or requested_session_id,
            candidates,
        )
        if catalog_resolution.session_id:
            return catalog_resolution
    if latest_session_id:
        return SessionResolution(latest_session_id, 0.9, "active")
    return _resolve_session_from_candidates(
        latest_user_message or requested_session_id,
        candidates,
    )


def _resolve_session_id_from_state(state: Mapping[str, object]) -> str:
    resolution = _resolve_session_from_state(state)
    return resolution.session_id or ""


def _resolve_session_id_from_catalog_output(user_prompt: str, tool_output: str) -> str:
    """Resolve a likely target session id from list-sessions output."""
    resolution = _resolve_session_from_candidates(
        user_prompt,
        parse_session_candidates_from_list_output(tool_output),
    )
    return resolution.session_id or ""


def resolve_session_id_from_list_output(user_prompt: str, tool_output: str) -> str:
    """Public compatibility wrapper for resolving a session id from catalog output."""
    return _resolve_session_id_from_catalog_output(user_prompt, tool_output)


def run_list_sessions_tool(cfg: AgentConfig) -> str:
    """Execute the list-sessions tool for LangGraph mode."""
    return list_sessions_impl(cfg)


def run_list_session_candidates_tool(cfg: AgentConfig) -> list[SessionCandidate]:
    """Execute the structured list-sessions helper for LangGraph mode."""
    return list_session_candidates_impl(cfg)


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


def _load_session_catalog(cfg: AgentConfig) -> tuple[list[SessionCandidate], str]:
    try:
        candidates = run_list_session_candidates_tool(cfg)
        return candidates, render_session_catalog(candidates)
    except Exception:
        tool_output = run_list_sessions_tool(cfg)
        return parse_session_candidates_from_list_output(tool_output), tool_output


def _bootstrap_session_catalog_if_needed(
    state: Mapping[str, object],
    cfg: AgentConfig,
) -> dict[str, Any]:
    """Populate session catalog output when it is missing."""
    current = ensure_langgraph_state(state)
    session_catalog_output = normalize_output(get_state_field(current, "session_catalog_output"))
    candidates = _session_candidates_from_state(current)
    if candidates and session_catalog_output.strip():
        return current
    if session_catalog_output.strip():
        current = set_state_fields(
            dict(current),
            session_candidates=candidates_to_state(candidates),
        )
    else:
        candidates, session_catalog_output = _load_session_catalog(cfg)
        current = set_state_fields(
            dict(current),
            session_catalog_output=session_catalog_output,
            session_candidates=candidates_to_state(candidates),
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
    resolution = _resolve_session_from_state(current)
    session_id = resolution.session_id or ""
    if session_id or not bootstrap_catalog:
        return current, session_id

    current = _bootstrap_session_catalog_if_needed(current, cfg)
    session_id = _resolve_session_id_from_state(current)
    return current, session_id


def _with_optional_span(
    tracer,
    span_name: str,
    current: dict[str, Any],
    trace_full_state: bool,
    fn: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    if tracer is None:
        return fn()

    with tracer.start_as_current_span(span_name) as span:
        if trace_full_state:
            span.set_attribute("state.before.json", serialize_langgraph_state(current))

        updated = fn()

        span.set_attribute("tool.name", normalize_output(get_state_field(updated, "tool_name")))
        span.set_attribute(
            "output.value", normalize_output(get_state_field(updated, "tool_output"))
        )

        session_id = normalize_output(get_state_field(updated, "latest_session_id")).strip()
        if session_id:
            span.set_attribute("session.id", session_id)

        if trace_full_state:
            span.set_attribute("state.after.json", serialize_langgraph_state(updated))

        return updated


async def _with_optional_span_async(
    tracer,
    span_name: str,
    current: dict[str, Any],
    trace_full_state: bool,
    fn: Callable[[], Awaitable[dict[str, Any]]],
) -> dict[str, Any]:
    if tracer is None:
        return await fn()

    with tracer.start_as_current_span(span_name) as span:
        if trace_full_state:
            span.set_attribute("state.before.json", serialize_langgraph_state(current))

        updated = await fn()

        span.set_attribute("tool.name", normalize_output(get_state_field(updated, "tool_name")))
        span.set_attribute(
            "output.value", normalize_output(get_state_field(updated, "tool_output"))
        )

        session_id = normalize_output(get_state_field(updated, "latest_session_id")).strip()
        if session_id:
            span.set_attribute("session.id", session_id)

        if trace_full_state:
            span.set_attribute("state.after.json", serialize_langgraph_state(updated))

        return updated


def build_tool_list_sessions_node(
    *,
    cfg: AgentConfig,
    tracer=None,
    trace_full_state: bool = False,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Build the LangGraph node for listing sessions."""

    def tool_list_sessions_node(state: dict[str, Any]) -> dict[str, Any]:
        current = ensure_langgraph_state(state)
        latest_user_message = normalize_output(get_state_field(current, "latest_user_message"))
        existing_session_id = normalize_output(
            get_state_field(current, "latest_session_id")
        ).strip()

        def run() -> dict[str, Any]:
            candidates, tool_output = _load_session_catalog(cfg)
            updated = set_state_fields(
                dict(current),
                tool_name="list_sessions",
                tool_output=tool_output,
                session_catalog_output=tool_output,
                session_candidates=candidates_to_state(candidates),
            )
            resolved_session_id = (
                _resolve_session_from_candidates(latest_user_message, candidates).session_id or ""
            )
            if resolved_session_id:
                updated = set_state_fields(updated, latest_session_id=resolved_session_id)
                if resolved_session_id != existing_session_id:
                    updated = set_state_fields(updated, latest_session_transcript="")
            return updated

        return _with_optional_span(
            tracer,
            "langgraph-tool-list-sessions",
            current,
            trace_full_state,
            run,
        )

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

        def run() -> dict[str, Any]:
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
                latest_session_transcript=tool_output,
            )

        return _with_optional_span(
            tracer,
            "langgraph-tool-query-conversation",
            current,
            trace_full_state,
            run,
        )

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

        async def run() -> dict[str, Any]:
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

        return await _with_optional_span_async(
            tracer,
            "langgraph-tool-analyze-summary",
            current,
            trace_full_state,
            run,
        )

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
        session_id = normalize_output(get_state_field(current, "latest_session_id")).strip()
        content = normalize_output(get_state_field(current, "latest_generated_content"))
        title = normalize_output(get_state_field(current, "latest_generated_title")).strip() or None

        def run() -> dict[str, Any]:
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

        return _with_optional_span(
            tracer,
            "langgraph-tool-save-to-siyuan",
            current,
            trace_full_state,
            run,
        )

    return tool_save_to_siyuan_node


def build_tool_memorize_node(
    *,
    cfg: AgentConfig,
    tracer=None,
    trace_full_state: bool = False,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Build the LangGraph node for saving a fact to persistent memory."""

    async def tool_memorize_node(state: dict[str, Any]) -> dict[str, Any]:
        from pawn_agent.tools.memorize import build as _build_memorize  # noqa: PLC0415

        current = ensure_langgraph_state(state)

        async def run() -> dict[str, Any]:
            latest_user_message = normalize_output(get_state_field(current, "latest_user_message"))
            fact = (
                normalize_output(get_state_field(current, "tool_output")).strip()
                or latest_user_message
            )

            tool = _build_memorize(cfg)
            try:
                tool_output = await tool.function(fact)  # type: ignore[attr-defined]
            except Exception as exc:
                tool_output = f"Error memorizing: {exc}"

            return set_state_fields(
                dict(current),
                tool_name="memorize",
                tool_output=tool_output,
            )

        return await _with_optional_span_async(
            tracer,
            "langgraph-tool-memorize",
            current,
            trace_full_state,
            run,
        )

    return tool_memorize_node


def build_tool_recall_memory_node(
    *,
    cfg: AgentConfig,
    tracer=None,
    trace_full_state: bool = False,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Build the LangGraph node for recalling facts from persistent memory."""

    async def tool_recall_memory_node(state: dict[str, Any]) -> dict[str, Any]:
        from pawn_agent.tools.recall_memory import build as _build_recall  # noqa: PLC0415

        current = ensure_langgraph_state(state)

        async def run() -> dict[str, Any]:
            query = normalize_output(get_state_field(current, "latest_user_message")).strip()
            tool = _build_recall(cfg)
            try:
                tool_output = await tool.function(query)  # type: ignore[attr-defined]
            except Exception as exc:
                tool_output = f"Error recalling memories: {exc}"

            return set_state_fields(
                dict(current),
                tool_name="recall_memory",
                tool_output=tool_output,
            )

        return await _with_optional_span_async(
            tracer,
            "langgraph-tool-recall-memory",
            current,
            trace_full_state,
            run,
        )

    return tool_recall_memory_node


def build_tool_search_knowledge_node(
    *,
    cfg: AgentConfig,
    tracer=None,
    trace_full_state: bool = False,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Build the LangGraph node for semantic search over the knowledge store."""

    async def tool_search_knowledge_node(state: dict[str, Any]) -> dict[str, Any]:
        current = ensure_langgraph_state(state)

        async def run() -> dict[str, Any]:
            query = normalize_output(get_state_field(current, "latest_user_message")).strip()
            session_id = (
                normalize_output(get_state_field(current, "latest_session_id")).strip() or None
            )

            try:
                tool_output = await search_knowledge_impl(cfg, query, session_id=session_id)
            except Exception as exc:
                tool_output = f"Error searching knowledge: {exc}"

            return set_state_fields(
                dict(current),
                tool_name="search_knowledge",
                tool_output=tool_output,
            )

        return await _with_optional_span_async(
            tracer,
            "langgraph-tool-search-knowledge",
            current,
            trace_full_state,
            run,
        )

    return tool_search_knowledge_node


def build_tool_vectorize_node(
    *,
    cfg: AgentConfig,
    tracer=None,
    trace_full_state: bool = False,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Build the LangGraph node for indexing a session or SiYuan page."""

    async def tool_vectorize_node(state: dict[str, Any]) -> dict[str, Any]:
        current = ensure_langgraph_state(state)

        async def run() -> dict[str, Any]:
            session_id = (
                normalize_output(get_state_field(current, "latest_session_id")).strip() or None
            )

            if not session_id:
                return set_state_fields(
                    dict(current),
                    tool_name="vectorize",
                    tool_output=(
                        "I need a session in focus before I can index it. "
                        "Retrieve or analyze a session first."
                    ),
                )

            try:
                tool_output = await vectorize_impl(cfg, session_id=session_id)
            except Exception as exc:
                tool_output = f"Error vectorizing session: {exc}"

            return set_state_fields(
                dict(current),
                tool_name="vectorize",
                tool_output=tool_output,
            )

        return await _with_optional_span_async(
            tracer,
            "langgraph-tool-vectorize",
            current,
            trace_full_state,
            run,
        )

    return tool_vectorize_node


def _build_progress_payload(state: Mapping[str, object]) -> dict[str, object]:
    """Build an auto-generated progress notification payload from LangGraph state."""
    completed = normalize_output(get_state_field(state, "tool_name")).strip()
    remaining = list(get_state_field(state, "action_plan") or [])
    session_id = normalize_output(get_state_field(state, "latest_session_id")).strip()
    latest_user_message = normalize_output(get_state_field(state, "latest_user_message")).strip()

    parts: list[str] = []
    if completed:
        parts.append(f"Completed: {completed}")
    if remaining:
        parts.append(f"Next: {', '.join(str(r) for r in remaining)}")
    if session_id:
        parts.append(f"Session: {session_id}")

    message = " | ".join(parts) if parts else "Working on your request..."
    payload: dict[str, object] = {
        "message": message,
        "completed_step": completed,
        "remaining_steps": remaining,
    }
    if session_id:
        payload["session_id"] = session_id
    if latest_user_message:
        payload["original_prompt"] = latest_user_message
    return payload


def _pick_notification_target(cfg: AgentConfig) -> str | None:
    """Return the best default notification target from queue_producers."""
    if not cfg.queue_producers:
        return None
    # Prefer a target named "matrix" if it exists.
    if "matrix" in cfg.queue_producers:
        return "matrix"
    return next(iter(cfg.queue_producers.keys()))


def _schedule_prompt_from_user_message(user_prompt: str, session_id: str) -> str:
    """Derive a useful scheduled prompt from the user's wording."""
    lowered = normalize_output(user_prompt).strip().lower()
    if "analy" in lowered and "siyuan" in lowered:
        return (
            f"Run the standard structured analysis for session {session_id} "
            "and save it to SiYuan."
        )
    if "analy" in lowered:
        return f"Run the standard structured analysis for session {session_id}."
    if "summar" in lowered and "siyuan" in lowered:
        return f"Summarize session {session_id} and save the result to SiYuan."
    if "siyuan" in lowered:
        return f"Review session {session_id} and save the result to SiYuan."
    return f"Analyze the latest conversation for session {session_id}."


def _parse_interval_seconds(user_prompt: str) -> int | None:
    normalized = normalize_output(user_prompt).strip()
    lowered = normalized.lower()

    match = _INTERVAL_RE.search(normalized)
    if match:
        value = int(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith(("hour", "hr")):
            return value * 3600
        return value * 60

    if "every half hour" in lowered or "each half hour" in lowered:
        return 1800
    if "every hour" in lowered or "each hour" in lowered or "hourly" in lowered:
        return 3600
    if "daily" in lowered:
        return 86400
    return None


def _infer_schedule_from_user_message(
    user_prompt: str,
    latest_session_id: str,
) -> dict[str, object] | None:
    """Best-effort deterministic fallback for common scheduling requests."""
    session_id = latest_session_id.strip()
    if not session_id:
        return None

    normalized = normalize_output(user_prompt).strip()
    lowered = normalized.lower()
    if not any(
        word in lowered for word in ("schedule", "every", "each", "hour", "minute", "scan", "daily")
    ):
        return None

    interval_seconds = _parse_interval_seconds(normalized)
    if interval_seconds is None:
        return None

    return {
        "action": "create",
        "name": "Recurring latest conversation analysis",
        "prompt": _schedule_prompt_from_user_message(normalized, session_id),
        "schedule": {
            "session_id": session_id,
            "schedule_kind": "interval",
            "interval_seconds": interval_seconds,
        },
        "rationale": normalized,
    }


def _normalize_schedule_tool_params(
    params: dict[str, object] | None,
    *,
    user_prompt: str,
    latest_session_id: str,
) -> dict[str, object] | None:
    """Fill obvious gaps in extracted schedule params before proposal creation."""
    resolved = dict(params or {})
    inferred = _infer_schedule_from_user_message(user_prompt, latest_session_id) or {}
    if not resolved:
        resolved = inferred
    schedule = resolved.get("schedule")
    if not isinstance(schedule, dict):
        schedule = {}
    else:
        schedule = dict(schedule)
    inferred_schedule = inferred.get("schedule")
    if isinstance(inferred_schedule, dict):
        for key, value in inferred_schedule.items():
            if key not in schedule or schedule.get(key) in (None, ""):
                schedule[key] = value

    if latest_session_id and not schedule.get("session_id"):
        if "latest conversation" in user_prompt.lower() or "latest session" in user_prompt.lower():
            schedule["session_id"] = latest_session_id
        elif resolved.get("action") == "create":
            schedule["session_id"] = latest_session_id

    if resolved.get("action") == "create":
        if not resolved.get("name"):
            resolved["name"] = inferred.get("name") or "Recurring latest conversation analysis"
        if not resolved.get("prompt") and schedule.get("session_id"):
            resolved["prompt"] = _schedule_prompt_from_user_message(
                user_prompt,
                str(schedule["session_id"]),
            )
        if not resolved.get("rationale") and inferred.get("rationale"):
            resolved["rationale"] = inferred["rationale"]

    if schedule:
        resolved["schedule"] = schedule
    return resolved or None


def build_tool_push_queue_message_node(
    *,
    cfg: AgentConfig,
    chat_agent,
    tracer=None,
    trace_full_state: bool = False,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Build the LangGraph node for sending progress notifications to a queue."""

    async def tool_push_queue_message_node(state: dict[str, Any]) -> dict[str, Any]:
        current = ensure_langgraph_state(state)

        async def run() -> dict[str, Any]:
            latest_user_message = normalize_output(
                get_state_field(current, "latest_user_message")
            ).strip()
            chat_history = get_recent_messages(current)

            valid_targets = list(cfg.queue_producers.keys()) if cfg.queue_producers else []
            target = _pick_notification_target(cfg)
            if not target:
                return set_state_fields(
                    dict(current),
                    tool_name="push_queue_message",
                    tool_output="Error: no queue_producers configured in pawnai.yaml.",
                )

            params = await chat_agent.extract_queue_publish_params(
                latest_user_message,
                chat_history,
                valid_targets,
            )
            if params:
                payload = dict(params.get("payload") or {})
                command = str(params.get("command") or "notify")
            else:
                command = "notify"
                payload = _build_progress_payload(current)

            try:
                tool_output = await push_queue_message_impl(
                    cfg,
                    target=target,
                    command=command,
                    payload=payload,
                )
            except Exception as exc:
                tool_output = f"Error pushing queue message: {exc}"

            return set_state_fields(
                dict(current),
                tool_name="push_queue_message",
                tool_output=tool_output,
            )

        return await _with_optional_span_async(
            tracer,
            "langgraph-tool-push-queue-message",
            current,
            trace_full_state,
            run,
        )

    return tool_push_queue_message_node


def build_tool_propose_schedule_change_node(
    *,
    cfg: AgentConfig,
    chat_agent,
    tracer=None,
    trace_full_state: bool = False,
) -> Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]:
    """Build the LangGraph node for creating schedule-change proposals."""

    async def tool_propose_schedule_change_node(state: dict[str, Any]) -> dict[str, Any]:
        current = ensure_langgraph_state(state)
        current, resolved_session_id = resolve_session_id(current, cfg)
        if resolved_session_id and (
            normalize_output(get_state_field(current, "latest_session_id")).strip()
            != resolved_session_id
        ):
            current = set_state_fields(dict(current), latest_session_id=resolved_session_id)
        latest_user_message = normalize_output(
            get_state_field(current, "latest_user_message")
        ).strip()
        latest_session_id = normalize_output(get_state_field(current, "latest_session_id")).strip()
        chat_history = get_recent_messages(current)

        async def run() -> dict[str, Any]:
            params = await chat_agent.extract_schedule_proposal_params(
                latest_user_message,
                chat_history,
                latest_session_id=latest_session_id,
            )
            params = _normalize_schedule_tool_params(
                params,
                user_prompt=latest_user_message,
                latest_session_id=latest_session_id,
            )
            if not params:
                tool_output = (
                    "I could not extract a valid schedule proposal. "
                    "Please provide the action, schedule timing, prompt, and session id."
                )
            else:
                schedule = params.get("schedule")
                if (
                    isinstance(schedule, dict)
                    and latest_session_id
                    and not schedule.get("session_id")
                ):
                    schedule = dict(schedule)
                    schedule["session_id"] = latest_session_id
                try:
                    tool_output = await propose_schedule_change_impl(
                        cfg,
                        action=str(params.get("action") or ""),
                        schedule_id=(
                            str(params.get("schedule_id")).strip()
                            if params.get("schedule_id")
                            else None
                        ),
                        name=str(params.get("name")).strip() if params.get("name") else None,
                        prompt=str(params.get("prompt")).strip() if params.get("prompt") else None,
                        schedule=schedule if isinstance(schedule, dict) else None,
                        timezone=(
                            str(params.get("timezone")).strip() if params.get("timezone") else None
                        ),
                        model=str(params.get("model")).strip() if params.get("model") else None,
                        rationale=(
                            str(params.get("rationale")).strip()
                            if params.get("rationale")
                            else latest_user_message
                        ),
                        proposed_by_session_id=latest_session_id or None,
                    )
                except Exception as exc:
                    tool_output = f"Error creating schedule proposal: {exc}"

            return set_state_fields(
                dict(current),
                tool_name="propose_schedule_change",
                tool_output=tool_output,
            )

        return await _with_optional_span_async(
            tracer,
            "langgraph-tool-propose-schedule-change",
            current,
            trace_full_state,
            run,
        )

    return tool_propose_schedule_change_node
