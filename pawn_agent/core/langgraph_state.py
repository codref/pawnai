"""Structured LangGraph state helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import json
from typing import Any, TypedDict

from pawn_agent.core.chat_primitives import normalize_output


RECENT_MESSAGE_LIMIT = 12


class LangGraphSessionState(TypedDict, total=False):
    incoming_prompt: str
    latest_user_message: str
    latest_assistant_message: str
    turn_count: int
    route_kind: str
    route_model: str
    reply_model: str
    tool_name: str
    requested_session_id: str
    action_plan: list[str]


class LangGraphDurableFacts(TypedDict, total=False):
    latest_session_id: str


class LangGraphArtifacts(TypedDict, total=False):
    tool_output: str
    latest_generated_content: str
    latest_generated_title: str
    session_catalog_output: str
    latest_session_transcript: str


class StructuredLangGraphState(TypedDict, total=False):
    session_state: LangGraphSessionState
    durable_facts: LangGraphDurableFacts
    artifacts: LangGraphArtifacts
    recent_messages: list[dict[str, str]]


SESSION_STATE_DEFAULTS: LangGraphSessionState = {
    "incoming_prompt": "",
    "latest_user_message": "",
    "latest_assistant_message": "",
    "turn_count": 0,
    "route_kind": "",
    "route_model": "",
    "reply_model": "",
    "tool_name": "",
    "requested_session_id": "",
    "action_plan": [],
}

DURABLE_FACT_DEFAULTS: LangGraphDurableFacts = {
    "latest_session_id": "",
}

ARTIFACT_DEFAULTS: LangGraphArtifacts = {
    "tool_output": "",
    "latest_generated_content": "",
    "latest_generated_title": "",
    "session_catalog_output": "",
    "latest_session_transcript": "",
}

FIELD_BUCKETS = {
    "incoming_prompt": "session_state",
    "latest_user_message": "session_state",
    "latest_assistant_message": "session_state",
    "turn_count": "session_state",
    "route_kind": "session_state",
    "route_model": "session_state",
    "reply_model": "session_state",
    "tool_name": "session_state",
    "requested_session_id": "session_state",
    "action_plan": "session_state",
    "latest_session_id": "durable_facts",
    "tool_output": "artifacts",
    "latest_generated_content": "artifacts",
    "latest_generated_title": "artifacts",
    "session_catalog_output": "artifacts",
    "latest_session_transcript": "artifacts",
}

FIELD_DEFAULTS = {
    **SESSION_STATE_DEFAULTS,
    **DURABLE_FACT_DEFAULTS,
    **ARTIFACT_DEFAULTS,
}


def new_structured_langgraph_state() -> dict[str, Any]:
    """Return the bucketed LangGraph state."""
    return {
        "session_state": dict(SESSION_STATE_DEFAULTS),
        "durable_facts": dict(DURABLE_FACT_DEFAULTS),
        "artifacts": dict(ARTIFACT_DEFAULTS),
        "recent_messages": [],
    }


def _copy_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    return {}


def _trim_recent_messages(messages: Sequence[Mapping[str, object]]) -> list[dict[str, str]]:
    trimmed: list[dict[str, str]] = []
    for item in list(messages)[-RECENT_MESSAGE_LIMIT:]:
        role = normalize_output(item.get("role", "")).strip()
        content = normalize_output(item.get("content", ""))
        if not role:
            continue
        trimmed.append({"role": role, "content": content})
    return trimmed


def ensure_langgraph_state(state: Mapping[str, object] | None) -> dict[str, Any]:
    """Normalize any partial state into the structured state shape."""
    original = dict(state or {})
    normalized = new_structured_langgraph_state()

    session_state = {
        **SESSION_STATE_DEFAULTS,
        **_copy_mapping(original.get("session_state")),
    }
    durable_facts = {
        **DURABLE_FACT_DEFAULTS,
        **_copy_mapping(original.get("durable_facts")),
    }
    artifacts = {
        **ARTIFACT_DEFAULTS,
        **_copy_mapping(original.get("artifacts")),
    }

    normalized["session_state"] = session_state
    normalized["durable_facts"] = durable_facts
    normalized["artifacts"] = artifacts
    normalized["recent_messages"] = _trim_recent_messages(original.get("recent_messages", []))
    return normalized


def get_state_field(state: Mapping[str, object], field: str) -> Any:
    normalized = ensure_langgraph_state(state)
    bucket = FIELD_BUCKETS[field]
    if bucket == "session_state":
        return normalized["session_state"].get(field, FIELD_DEFAULTS[field])
    if bucket == "durable_facts":
        return normalized["durable_facts"].get(field, FIELD_DEFAULTS[field])
    return normalized["artifacts"].get(field, FIELD_DEFAULTS[field])


def set_state_field(state: dict[str, Any], field: str, value: Any) -> dict[str, Any]:
    normalized = ensure_langgraph_state(state)
    bucket = FIELD_BUCKETS.get(field)
    if bucket == "session_state":
        normalized["session_state"][field] = value
    elif bucket == "durable_facts":
        normalized["durable_facts"][field] = value
    elif bucket == "artifacts":
        normalized["artifacts"][field] = value
    return normalized


def set_state_fields(state: dict[str, Any], **values: Any) -> dict[str, Any]:
    normalized = ensure_langgraph_state(state)
    for field, value in values.items():
        normalized = set_state_field(normalized, field, value)
    return normalized


def get_recent_messages(state: Mapping[str, object]) -> list[dict[str, str]]:
    normalized = ensure_langgraph_state(state)
    return list(normalized.get("recent_messages", []))


def set_recent_messages(
    state: dict[str, Any],
    messages: Sequence[Mapping[str, object]],
) -> dict[str, Any]:
    normalized = ensure_langgraph_state(state)
    trimmed = _trim_recent_messages(messages)
    normalized["recent_messages"] = trimmed
    return normalized


def serialize_langgraph_state(state: Mapping[str, object] | None) -> str:
    """Return the full normalized LangGraph state as JSON for tracing."""
    normalized = ensure_langgraph_state(state)
    return json.dumps(normalized, ensure_ascii=False, sort_keys=True, default=str)
