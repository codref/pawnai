"""Minimal LangGraph-managed chat path for evaluating dynamic context orchestration."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Callable, TypedDict

from prompt_toolkit import PromptSession

from pawn_agent.core.chat_primitives import (
    PlainPydanticChatAgent,
    apply_assistant_message,
    apply_user_message,
    normalize_output,
)
from pawn_agent.core.langgraph_state import (
    StructuredLangGraphState,
    ensure_langgraph_state,
    get_recent_messages,
    get_state_field,
    new_structured_langgraph_state,
    serialize_langgraph_state,
    set_recent_messages,
    set_state_fields,
)
from pawn_agent.core.langgraph_tools import (
    build_tool_analyze_summary_node,
    build_tool_list_sessions_node,
    build_tool_memorize_node,
    build_tool_push_queue_message_node,
    build_tool_query_conversation_node,
    build_tool_recall_memory_node,
    build_tool_save_to_siyuan_node,
    build_tool_search_knowledge_node,
    build_tool_vectorize_node,
)
from pawn_agent.utils.config import AgentConfig


class LangGraphChatState(StructuredLangGraphState, total=False):
    """State carried through the minimal LangGraph chatbot."""


def new_langgraph_chat_state() -> LangGraphChatState:
    """Return the default LangGraph chat state."""
    return ensure_langgraph_state(new_structured_langgraph_state())


def _import_langgraph_core():
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - exercised via CLI/runtime integration
        raise RuntimeError(
            "LangGraph chat requires the optional 'langgraph' dependency to be installed."
        ) from exc
    return StateGraph, START, END


def _import_phoenix_register():
    try:
        from phoenix.otel import register
    except ImportError as exc:  # pragma: no cover - exercised via CLI/runtime integration
        raise RuntimeError(
            "LangGraph chat Phoenix tracing requires the optional 'arize-phoenix-otel' dependency."
        ) from exc
    return register


def _import_trace_api():
    try:
        from opentelemetry import trace
    except ImportError as exc:  # pragma: no cover - exercised via CLI/runtime integration
        raise RuntimeError(
            "LangGraph chat Phoenix tracing requires OpenTelemetry support to be installed."
        ) from exc
    return trace


def build_phoenix_tracer(cfg: AgentConfig):
    """Build a Phoenix-backed tracer when tracing is enabled."""
    if not cfg.phoenix_enabled:
        return None

    register = _import_phoenix_register()
    trace = _import_trace_api()
    kwargs: dict[str, object] = {}
    if cfg.phoenix_api_key:
        kwargs["headers"] = {"authorization": f"Bearer {cfg.phoenix_api_key}"}
    register(
        project_name=cfg.phoenix_project_name,
        endpoint=cfg.phoenix_endpoint,
        protocol=cfg.phoenix_protocol,
        auto_instrument=False,
        batch=True,
        **kwargs,
    )
    return trace.get_tracer(__name__)


def _trace_full_state_snapshot(span, state: LangGraphChatState, *, label: str) -> None:
    span.set_attribute(f"state.{label}.json", serialize_langgraph_state(state))


class LangGraphRouterChatAgent:
    """Forward-only fast/deep router used by the LangGraph evaluation path."""

    VALID_ACTIONS = {
        "reply_fast",
        "reply_deep",
        "tool_list_sessions",
        "tool_analyze_summary",
        "tool_query_conversation",
        "tool_save_to_siyuan",
        "tool_memorize",
        "tool_recall_memory",
        "tool_search_knowledge",
        "tool_vectorize",
        "tool_push_queue_message",
    }

    def __init__(
        self,
        cfg: AgentConfig,
        on_thinking: Callable[[], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self._on_thinking = on_thinking
        self._fast_cfg = self._clone_cfg_with_model(cfg.langgraph_fast_model)
        self._deep_cfg = self._clone_cfg_with_model(cfg.langgraph_deep_model)
        self._fast_agent = PlainPydanticChatAgent(cfg=self._fast_cfg, on_thinking=None)
        self._deep_agent = PlainPydanticChatAgent(cfg=self._deep_cfg, on_thinking=None)
        self.model_name = self.cfg.langgraph_fast_model
        self.last_route_kind = ""
        self.last_route_model = self.cfg.langgraph_fast_model
        self.last_reply_model = ""
        self.last_tool_name = ""
        self.last_action_plan: list[str] = []

    def _clone_cfg_with_model(self, model_name: str) -> AgentConfig:
        cloned = (
            self.cfg.model_copy(deep=True)
            if hasattr(self.cfg, "model_copy")
            else deepcopy(self.cfg)
        )
        cloned.pydantic_model = model_name
        return cloned

    def _history_for_prompt(
        self,
        chat_history: list[dict[str, str]],
        user_prompt: str,
    ) -> list[dict[str, str]]:
        prior_history = list(chat_history)
        if (
            prior_history
            and prior_history[-1].get("role") == "user"
            and prior_history[-1].get("content") == user_prompt
        ):
            prior_history = prior_history[:-1]
        return prior_history

    def _history_transcript(
        self,
        chat_history: list[dict[str, str]],
        user_prompt: str,
    ) -> str:
        prior_history = self._history_for_prompt(chat_history, user_prompt)
        if not prior_history:
            return "(no prior conversation)"
        lines: list[str] = []
        for item in prior_history:
            role = normalize_output(item.get("role", "")).strip().lower()
            content = normalize_output(item.get("content", "")).strip()
            if not content:
                continue
            speaker = "User" if role == "user" else "Assistant"
            lines.append(f"{speaker}: {content}")
        return "\n".join(lines) if lines else "(no prior conversation)"

    def _planner_prompt(
        self,
        user_prompt: str,
        chat_history: list[dict[str, str]],
        *,
        latest_session_id: str = "",
        latest_generated_content: str = "",
        latest_session_transcript: str = "",
        recent_memories: list[str] | None = None,
    ) -> str:
        session_focus = latest_session_id or "(none)"
        has_generated_content = (
            "yes" if normalize_output(latest_generated_content).strip() else "no"
        )
        has_session_transcript = (
            "yes" if normalize_output(latest_session_transcript).strip() else "no"
        )
        memories_section = ""
        if recent_memories:
            memory_lines = "\n".join(f"- {m}" for m in recent_memories)
            memories_section = "\nRelevant memories from previous sessions:\n" f"{memory_lines}\n"
        return (
            "You are a planning model for a LangGraph chat system.\n"
            "Decompose the user's request into an ordered list of actions and return it as a JSON array of strings.\n"
            "Return ONLY the JSON array — no explanation, no markdown, no other text.\n\n"
            "Available actions:\n"
            "- reply_fast: simple conversational reply, lightweight reasoning, or normal follow-up\n"
            "- reply_deep: thorough analysis, synthesis, executive reports, comprehensive summaries, or complex reasoning\n"
            "- tool_list_sessions: discover available sessions when none is in focus\n"
            "- tool_analyze_summary: produce the fixed structured analysis (Title / Summary / Key Topics / Speaker Highlights / Sentiment / Tags). This tool loads the transcript itself.\n"
            "- tool_query_conversation: load a conversation transcript. Include this before reply_fast/reply_deep when the user asks for a report or summary about a session AND the session transcript is not already cached.\n"
            "- tool_save_to_siyuan: Use ONLY when the user explicitly asks to save or export something to SiYuan.\n"
            "- tool_memorize: Use when the user explicitly asks to remember, save, or memorize a fact or preference.\n"
            "- tool_recall_memory: Search persistent memory for facts the user previously asked to remember.\n"
            "- tool_search_knowledge: Semantic search over indexed session chunks and SiYuan notes.\n"
            "- tool_vectorize: Index a session or SiYuan page into the knowledge store for future search.\n"
            "- tool_push_queue_message: Send a progress / notification message to an external queue (e.g. Matrix). "
            "Use when the user asks to be kept posted, alerted, notified, or when they say 'keep me updated'. "
            "You MAY include this action MULTIPLE TIMES in the same plan to report progress after key steps.\n\n"
            "Rules:\n"
            "- Every plan must end with reply_fast or reply_deep.\n"
            "- Use reply_deep when the user asks for analysis, executive reports, comprehensive summaries, or complex reasoning.\n"
            "- Use reply_fast for conversational replies, simple follow-ups, or when just reporting a tool result.\n"
            "- If session_transcript_cached is 'yes', skip tool_query_conversation — the transcript is already available for the current session.\n"
            "- Include tool_save_to_siyuan ONLY when the user explicitly asks to save or export to SiYuan. Never include it speculatively after generating content.\n"
            "- Include tool_save_to_siyuan AFTER the reply step that produces the content, and BEFORE END.\n"
            "- If the user asks to save a specific earlier piece of content (not the latest), use reply_deep first to extract and reproduce it from the conversation history, then tool_save_to_siyuan.\n"
            "- Do NOT include extract_session_id — it is handled automatically.\n"
            "- Include each action at most once, EXCEPT tool_push_queue_message which may appear multiple times to report progress.\n"
            "- Maximum 8 actions.\n\n"
            "Examples:\n"
            '  User asks a simple question → ["reply_fast"]\n'
            '  User asks for an analysis → ["reply_deep"]\n'
            '  User asks to list sessions → ["tool_list_sessions", "reply_fast"]\n'
            '  User asks to retrieve and summarize a session (transcript not cached) → ["tool_query_conversation", "reply_deep"]\n'
            '  User asks a follow-up about an already-loaded session (transcript cached) → ["reply_deep"]\n'
            '  User asks to write a draft or recap → ["reply_deep"]\n'
            '  User asks to retrieve, write a report and save it to SiYuan → ["tool_query_conversation", "reply_deep", "tool_save_to_siyuan", "reply_fast"]\n'
            '  User asks to save the latest generated content → ["tool_save_to_siyuan", "reply_fast"]\n'
            '  User asks to save a specific earlier analysis or piece of content → ["reply_deep", "tool_save_to_siyuan", "reply_fast"]\n'
            '  User says \'remember that...\' → ["tool_memorize", "reply_fast"]\n'
            '  User asks about a fact they may have mentioned before → ["tool_recall_memory", "reply_fast"]\n'
            '  User asks to search across sessions → ["tool_search_knowledge", "reply_deep"]\n'
            '  User asks to index a session for search → ["tool_vectorize", "reply_fast"]\n'
            '  User asks for a deep analysis and to keep them posted → ["tool_push_queue_message", "tool_query_conversation", "tool_push_queue_message", "reply_deep", "tool_push_queue_message", "reply_fast"]\n\n'
            "Latest session in focus:\n"
            f"{session_focus}\n\n"
            "Session transcript cached:\n"
            f"{has_session_transcript}\n\n"
            "Latest generated content available (for context only — do NOT auto-save):\n"
            f"{has_generated_content}\n"
            f"{memories_section}\n"
            "Conversation so far:\n"
            f"{self._history_transcript(chat_history, user_prompt)}\n\n"
            "Current user message:\n"
            f"{user_prompt}\n"
        )

    def _session_id_extraction_prompt(
        self,
        user_prompt: str,
        chat_history: list[dict[str, str]],
        *,
        latest_session_id: str = "",
    ) -> str:
        session_focus = latest_session_id or "(none)"
        return (
            "You extract a session id for a LangGraph tool call.\n"
            "Return exactly one line.\n"
            "If the current user message explicitly identifies a session id, return only that session id.\n"
            "If the current user message does not provide a new explicit session id, return NONE.\n"
            "Do not invent ids. Do not explain. Do not reuse the latest session automatically.\n\n"
            "Latest session in focus:\n"
            f"{session_focus}\n\n"
            "Conversation so far:\n"
            f"{self._history_transcript(chat_history, user_prompt)}\n\n"
            "Current user message:\n"
            f"{user_prompt}\n"
        )

    def _normalize_plan(self, plan_reply: str) -> list[str]:
        MAX_PLAN_LENGTH = 8
        raw = normalize_output(plan_reply).strip() if plan_reply else ""
        # Try direct JSON parse first
        parsed: list[str] = []
        try:
            parsed = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            # Fallback: extract first JSON array from the reply
            match = re.search(r"\[.*?\]", raw, re.DOTALL)
            if match:
                try:
                    parsed = json.loads(match.group(0))
                except (json.JSONDecodeError, ValueError):
                    parsed = []
        if not isinstance(parsed, list):
            return ["reply_deep"]
        validated = [
            item for item in parsed if isinstance(item, str) and item in self.VALID_ACTIONS
        ][:MAX_PLAN_LENGTH]
        return validated if validated else ["reply_deep"]

    def _normalize_requested_session_id(self, reply: str) -> str:
        value = normalize_output(reply).strip().splitlines()[0].strip() if reply else ""
        if not value or value.upper() == "NONE":
            return ""
        return value

    async def extract_requested_session_id(
        self,
        user_prompt: str,
        chat_history: list[dict[str, str]],
        *,
        latest_session_id: str = "",
    ) -> str:
        extraction_reply = await self._fast_agent.reply(
            self._session_id_extraction_prompt(
                user_prompt,
                chat_history,
                latest_session_id=latest_session_id,
            ),
            [],
        )
        return self._normalize_requested_session_id(extraction_reply)

    def _queue_publish_extraction_prompt(
        self,
        user_prompt: str,
        chat_history: list[dict[str, str]],
        valid_targets: list[str],
    ) -> str:
        targets_str = ", ".join(valid_targets)
        return (
            "You extract queue-publish parameters from a user request.\n"
            "Return ONLY a JSON object with keys: target, command, payload.\n"
            f"target must be one of: {targets_str}.\n"
            "command is a string like 'run'.\n"
            "payload is a JSON object (can be empty {}).\n"
            "Do not explain. Return only the JSON object.\n\n"
            "Conversation so far:\n"
            f"{self._history_transcript(chat_history, user_prompt)}\n\n"
            "Current user message:\n"
            f"{user_prompt}\n"
        )

    def _normalize_queue_publish_params(
        self, reply: str, valid_targets: list[str]
    ) -> dict[str, object] | None:
        raw = normalize_output(reply).strip() if reply else ""
        if not raw:
            return None
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            # Fallback: extract first JSON object from reply
            match = re.search(r"\{.*?\}", raw, re.DOTALL)
            if not match:
                return None
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                return None
        if not isinstance(parsed, dict):
            return None
        target = str(parsed.get("target", "")).strip()
        command = str(parsed.get("command", "")).strip()
        payload = parsed.get("payload")
        if not isinstance(payload, dict):
            payload = {}
        if target not in valid_targets:
            return None
        if not command:
            return None
        return {"target": target, "command": command, "payload": payload}

    async def extract_queue_publish_params(
        self,
        user_prompt: str,
        chat_history: list[dict[str, str]],
        valid_targets: list[str],
    ) -> dict[str, object] | None:
        extraction_reply = await self._fast_agent.reply(
            self._queue_publish_extraction_prompt(
                user_prompt,
                chat_history,
                valid_targets,
            ),
            [],
        )
        return self._normalize_queue_publish_params(extraction_reply, valid_targets)

    async def plan(
        self,
        user_prompt: str,
        chat_history: list[dict[str, str]],
        *,
        latest_session_id: str = "",
        latest_generated_content: str = "",
        latest_session_transcript: str = "",
        state: dict | None = None,
    ) -> list[str]:
        if self._on_thinking is not None:
            self._on_thinking()

        self.last_route_model = self.cfg.langgraph_fast_model
        self.last_reply_model = ""
        self.last_tool_name = ""
        self.model_name = self.cfg.langgraph_fast_model
        plan_reply = await self._fast_agent.reply(
            self._planner_prompt(
                user_prompt,
                chat_history,
                latest_session_id=latest_session_id,
                latest_generated_content=latest_generated_content,
                latest_session_transcript=latest_session_transcript,
                recent_memories=get_state_field(state, "recent_memories") if state else [],
            ),
            [],
        )
        self.last_action_plan = self._normalize_plan(plan_reply)
        self.last_route_kind = self.last_action_plan[0] if self.last_action_plan else ""
        return self.last_action_plan

    async def respond_fast(
        self,
        user_prompt: str,
        chat_history: list[dict[str, str]],
        *,
        tool_output: str = "",
    ) -> str:
        self.last_reply_model = self.cfg.langgraph_fast_model
        self.model_name = self.last_reply_model
        if tool_output:
            prompt = (
                "Answer the user's request using the tool result below.\n"
                "Be concise and conversational. If the tool result is sufficient, do not invent extra facts.\n\n"
                "Tool result:\n"
                f"{tool_output}\n\n"
                "Current user message:\n"
                f"{user_prompt}"
            )
            return await self._fast_agent.reply(prompt, chat_history)
        return await self._fast_agent.reply(user_prompt, chat_history)

    async def respond_deep(
        self,
        user_prompt: str,
        chat_history: list[dict[str, str]],
        *,
        tool_output: str = "",
    ) -> str:
        self.last_reply_model = self.cfg.langgraph_deep_model
        self.model_name = self.last_reply_model
        if tool_output:
            prompt = (
                "Answer the user's request using the tool result below.\n"
                "Provide a thorough, well-structured response grounded in the tool result.\n"
                "Do not invent facts beyond what the tool result supports.\n\n"
                "Tool result:\n"
                f"{tool_output}\n\n"
                "Current user message:\n"
                f"{user_prompt}"
            )
            return await self._deep_agent.reply(prompt, chat_history)
        return await self._deep_agent.reply(user_prompt, chat_history)


_HEADING_RE = re.compile(r"^\s*#\s+(.+?)\s*$", re.MULTILINE)


def _infer_generated_title(content: str) -> str:
    match = _HEADING_RE.search(normalize_output(content))
    return match.group(1).strip() if match else ""


def _should_capture_generated_content(state: LangGraphChatState) -> bool:
    return normalize_output(get_state_field(state, "tool_name")).strip() != "save_to_siyuan"


def _next_node_after_extract_session_id(state: LangGraphChatState) -> str:
    route_kind = normalize_output(get_state_field(state, "route_kind")).strip()
    if route_kind == "tool_analyze_summary":
        return "tool_analyze_summary"
    return "tool_query_conversation"


def _next_node_from_dispatch(state: LangGraphChatState) -> str:
    route_kind = normalize_output(get_state_field(state, "route_kind")).strip()
    if not route_kind:
        return "__end__"
    if route_kind in {"tool_analyze_summary", "tool_query_conversation"}:
        return "extract_session_id"
    if route_kind == "tool_list_sessions":
        return "tool_list_sessions"
    if route_kind == "tool_save_to_siyuan":
        return "tool_save_to_siyuan"
    if route_kind == "tool_memorize":
        return "tool_memorize"
    if route_kind == "tool_recall_memory":
        return "tool_recall_memory"
    if route_kind == "tool_search_knowledge":
        return "tool_search_knowledge"
    if route_kind == "tool_vectorize":
        return "tool_vectorize"
    if route_kind == "tool_push_queue_message":
        return "tool_push_queue_message"
    if route_kind == "reply_fast":
        return "respond_fast"
    return "respond_deep"


async def build_langgraph_chat_graph(
    chat_agent: LangGraphRouterChatAgent,
    tracer=None,
    *,
    trace_full_state: bool = False,
):
    """Build the minimal LangGraph chatbot graph."""
    StateGraph, START, END = _import_langgraph_core()

    async def recall_memories_node(state: LangGraphChatState) -> LangGraphChatState:
        """Silently fetch relevant memories before planning."""
        from pawn_agent.core.store import NS_MEMORIES, get_store  # noqa: PLC0415

        _MEMORY_SCORE_THRESHOLD = 0.5
        current = ensure_langgraph_state(state)
        user_msg = normalize_output(get_state_field(current, "latest_user_message")).strip()
        if not user_msg:
            return current
        try:
            store = await get_store(chat_agent.cfg)
            results = await store.asearch(NS_MEMORIES, query=user_msg, limit=5)
            memories = [
                (item.value or {}).get("text", "").strip()
                for item in results
                if (item.score or 0.0) >= _MEMORY_SCORE_THRESHOLD
                and (item.value or {}).get("text", "").strip()
            ]
        except Exception:
            memories = []
        return set_state_fields(dict(current), recent_memories=memories)

    def human_input_node(state: LangGraphChatState) -> LangGraphChatState:
        current = ensure_langgraph_state(state)
        message_state = {
            "chat_history": get_recent_messages(current),
            "turn_count": int(get_state_field(current, "turn_count")),
        }
        if tracer is None:
            prompt = normalize_output(get_state_field(current, "incoming_prompt"))
            message_update = apply_user_message(message_state, prompt)
            updated = set_recent_messages(dict(current), message_update.get("chat_history", []))
            updated = set_state_fields(
                updated,
                incoming_prompt=prompt,
                latest_user_message=message_update.get("latest_user_message", ""),
                turn_count=int(message_update.get("turn_count", 0)),
            )
            return updated
        with tracer.start_as_current_span("langgraph-human-input") as span:
            if trace_full_state:
                _trace_full_state_snapshot(span, current, label="before")
            prompt = normalize_output(get_state_field(current, "incoming_prompt"))
            span.set_attribute("input.value", prompt)
            message_update = apply_user_message(message_state, prompt)
            updated = set_recent_messages(dict(current), message_update.get("chat_history", []))
            updated = set_state_fields(
                updated,
                incoming_prompt=prompt,
                latest_user_message=message_update.get("latest_user_message", ""),
                turn_count=int(message_update.get("turn_count", 0)),
            )
            span.set_attribute("output.turn_count", int(get_state_field(updated, "turn_count")))
            if trace_full_state:
                _trace_full_state_snapshot(span, updated, label="after")
            return updated

    async def plan_node(state: LangGraphChatState) -> LangGraphChatState:
        current = ensure_langgraph_state(state)
        latest_user_message = normalize_output(get_state_field(current, "latest_user_message"))
        chat_history = get_recent_messages(current)
        latest_session_id = normalize_output(get_state_field(current, "latest_session_id")).strip()
        latest_generated_content = normalize_output(
            get_state_field(current, "latest_generated_content")
        )
        latest_session_transcript = normalize_output(
            get_state_field(current, "latest_session_transcript")
        )
        if tracer is None:
            action_plan = await chat_agent.plan(
                latest_user_message,
                chat_history,
                latest_session_id=latest_session_id,
                latest_generated_content=latest_generated_content,
                latest_session_transcript=latest_session_transcript,
                state=current,
            )
            return set_state_fields(
                dict(current),
                route_kind="",
                route_model=chat_agent.last_route_model,
                action_plan=action_plan,
                tool_name="",
                tool_output="",
                requested_session_id="",
            )
        with tracer.start_as_current_span("langgraph-plan") as span:
            if trace_full_state:
                _trace_full_state_snapshot(span, current, label="before")
            action_plan = await chat_agent.plan(
                latest_user_message,
                chat_history,
                latest_session_id=latest_session_id,
                latest_generated_content=latest_generated_content,
                latest_session_transcript=latest_session_transcript,
                state=current,
            )
            updated = set_state_fields(
                dict(current),
                route_kind="",
                route_model=chat_agent.last_route_model,
                action_plan=action_plan,
                tool_name="",
                tool_output="",
                requested_session_id="",
            )
            span.set_attribute("llm.model_name", chat_agent.last_route_model)
            span.set_attribute("input.value", latest_user_message)
            span.set_attribute("output.action_plan", json.dumps(action_plan))
            if trace_full_state:
                _trace_full_state_snapshot(span, updated, label="after")
            return updated

    async def extract_session_id_node(state: LangGraphChatState) -> LangGraphChatState:
        current = ensure_langgraph_state(state)
        latest_user_message = normalize_output(get_state_field(current, "latest_user_message"))
        chat_history = get_recent_messages(current)
        latest_session_id = normalize_output(get_state_field(current, "latest_session_id")).strip()
        if tracer is None:
            requested_session_id = await chat_agent.extract_requested_session_id(
                latest_user_message,
                chat_history,
                latest_session_id=latest_session_id,
            )
            return set_state_fields(dict(current), requested_session_id=requested_session_id)
        with tracer.start_as_current_span("langgraph-extract-session-id") as span:
            if trace_full_state:
                _trace_full_state_snapshot(span, current, label="before")
            requested_session_id = await chat_agent.extract_requested_session_id(
                latest_user_message,
                chat_history,
                latest_session_id=latest_session_id,
            )
            updated = set_state_fields(dict(current), requested_session_id=requested_session_id)
            span.set_attribute("llm.model_name", chat_agent.last_route_model)
            span.set_attribute("input.value", latest_user_message)
            if latest_session_id:
                span.set_attribute("session.id", latest_session_id)
            span.set_attribute("output.requested_session_id", requested_session_id or "NONE")
            if trace_full_state:
                _trace_full_state_snapshot(span, updated, label="after")
            return updated

    def dispatch_node(state: LangGraphChatState) -> LangGraphChatState:
        current = ensure_langgraph_state(state)
        action_plan = list(get_state_field(current, "action_plan") or [])
        if tracer is None:
            if action_plan:
                route_kind = action_plan[0]
                remaining = action_plan[1:]
            else:
                route_kind = ""
                remaining = []
            return set_state_fields(dict(current), route_kind=route_kind, action_plan=remaining)
        with tracer.start_as_current_span("langgraph-dispatch") as span:
            if trace_full_state:
                _trace_full_state_snapshot(span, current, label="before")
            span.set_attribute("input.action_plan", json.dumps(action_plan))
            if action_plan:
                route_kind = action_plan[0]
                remaining = action_plan[1:]
            else:
                route_kind = ""
                remaining = []
            updated = set_state_fields(dict(current), route_kind=route_kind, action_plan=remaining)
            span.set_attribute("output.route_kind", route_kind or "__end__")
            if trace_full_state:
                _trace_full_state_snapshot(span, updated, label="after")
            return updated

    async def respond_fast_node(state: LangGraphChatState) -> LangGraphChatState:
        current = ensure_langgraph_state(state)
        latest_user_message = normalize_output(get_state_field(current, "latest_user_message"))
        chat_history = get_recent_messages(current)
        tool_output = normalize_output(get_state_field(current, "tool_output"))
        if not tool_output.strip():
            tool_output = normalize_output(get_state_field(current, "latest_session_transcript"))
        message_state = {
            "chat_history": chat_history,
        }
        if tracer is None:
            reply = await chat_agent.respond_fast(
                latest_user_message,
                chat_history,
                tool_output=tool_output,
            )
            message_update = apply_assistant_message(message_state, reply)
            updated = set_recent_messages(dict(current), message_update.get("chat_history", []))
            updated = set_state_fields(
                updated,
                incoming_prompt="",
                reply_model=chat_agent.last_reply_model,
                tool_output="",
                requested_session_id="",
                latest_assistant_message=reply,
            )
            if _should_capture_generated_content(state):
                updated = set_state_fields(
                    updated,
                    latest_generated_content=reply,
                    latest_generated_title=_infer_generated_title(reply),
                )
            return updated
        with tracer.start_as_current_span("langgraph-respond-fast") as span:
            if trace_full_state:
                _trace_full_state_snapshot(span, current, label="before")
            reply = await chat_agent.respond_fast(
                latest_user_message,
                chat_history,
                tool_output=tool_output,
            )
            message_update = apply_assistant_message(message_state, reply)
            updated = set_recent_messages(dict(current), message_update.get("chat_history", []))
            updated = set_state_fields(
                updated,
                incoming_prompt="",
                reply_model=chat_agent.last_reply_model,
                tool_output="",
                requested_session_id="",
                latest_assistant_message=reply,
            )
            if _should_capture_generated_content(state):
                updated = set_state_fields(
                    updated,
                    latest_generated_content=reply,
                    latest_generated_title=_infer_generated_title(reply),
                )
            span.set_attribute("llm.model_name", chat_agent.last_reply_model)
            span.set_attribute("input.value", latest_user_message)
            if normalize_output(get_state_field(current, "tool_name")):
                span.set_attribute(
                    "tool.name", normalize_output(get_state_field(current, "tool_name"))
                )
            if normalize_output(get_state_field(current, "latest_session_id")):
                span.set_attribute(
                    "session.id",
                    normalize_output(get_state_field(current, "latest_session_id")),
                )
            span.set_attribute("output.value", reply)
            if trace_full_state:
                _trace_full_state_snapshot(span, updated, label="after")
            return updated

    async def respond_deep_node(state: LangGraphChatState) -> LangGraphChatState:
        current = ensure_langgraph_state(state)
        latest_user_message = normalize_output(get_state_field(current, "latest_user_message"))
        chat_history = get_recent_messages(current)
        tool_output = normalize_output(get_state_field(current, "tool_output"))
        if not tool_output.strip():
            tool_output = normalize_output(get_state_field(current, "latest_session_transcript"))
        message_state = {
            "chat_history": chat_history,
        }
        if tracer is None:
            reply = await chat_agent.respond_deep(
                latest_user_message,
                chat_history,
                tool_output=tool_output,
            )
            message_update = apply_assistant_message(message_state, reply)
            updated = set_recent_messages(dict(current), message_update.get("chat_history", []))
            updated = set_state_fields(
                updated,
                incoming_prompt="",
                reply_model=chat_agent.last_reply_model,
                tool_output="",
                requested_session_id="",
                latest_assistant_message=reply,
            )
            if _should_capture_generated_content(state):
                updated = set_state_fields(
                    updated,
                    latest_generated_content=reply,
                    latest_generated_title=_infer_generated_title(reply),
                )
            return updated
        with tracer.start_as_current_span("langgraph-respond-deep") as span:
            if trace_full_state:
                _trace_full_state_snapshot(span, current, label="before")
            reply = await chat_agent.respond_deep(
                latest_user_message,
                chat_history,
                tool_output=tool_output,
            )
            message_update = apply_assistant_message(message_state, reply)
            updated = set_recent_messages(dict(current), message_update.get("chat_history", []))
            updated = set_state_fields(
                updated,
                incoming_prompt="",
                reply_model=chat_agent.last_reply_model,
                tool_output="",
                requested_session_id="",
                latest_assistant_message=reply,
            )
            if _should_capture_generated_content(state):
                updated = set_state_fields(
                    updated,
                    latest_generated_content=reply,
                    latest_generated_title=_infer_generated_title(reply),
                )
            span.set_attribute("llm.model_name", chat_agent.last_reply_model)
            span.set_attribute("input.value", latest_user_message)
            if normalize_output(get_state_field(current, "latest_session_id")):
                span.set_attribute(
                    "session.id",
                    normalize_output(get_state_field(current, "latest_session_id")),
                )
            span.set_attribute("output.value", reply)
            if trace_full_state:
                _trace_full_state_snapshot(span, updated, label="after")
            return updated

    builder = StateGraph(LangGraphChatState)
    builder.add_node("human_input", human_input_node)
    builder.add_node("recall_memories", recall_memories_node)
    builder.add_node("plan", plan_node)
    builder.add_node("dispatch", dispatch_node)
    builder.add_node("extract_session_id", extract_session_id_node)
    builder.add_node(
        "tool_list_sessions",
        build_tool_list_sessions_node(
            cfg=chat_agent.cfg,
            tracer=tracer,
            trace_full_state=trace_full_state,
        ),
    )
    builder.add_node(
        "tool_analyze_summary",
        build_tool_analyze_summary_node(
            cfg=chat_agent.cfg,
            tracer=tracer,
            trace_full_state=trace_full_state,
        ),
    )
    builder.add_node(
        "tool_query_conversation",
        build_tool_query_conversation_node(
            cfg=chat_agent.cfg,
            tracer=tracer,
            trace_full_state=trace_full_state,
        ),
    )
    builder.add_node(
        "tool_save_to_siyuan",
        build_tool_save_to_siyuan_node(
            cfg=chat_agent.cfg,
            tracer=tracer,
            trace_full_state=trace_full_state,
        ),
    )
    builder.add_node(
        "tool_memorize",
        build_tool_memorize_node(
            cfg=chat_agent.cfg,
            tracer=tracer,
            trace_full_state=trace_full_state,
        ),
    )
    builder.add_node(
        "tool_recall_memory",
        build_tool_recall_memory_node(
            cfg=chat_agent.cfg,
            tracer=tracer,
            trace_full_state=trace_full_state,
        ),
    )
    builder.add_node(
        "tool_search_knowledge",
        build_tool_search_knowledge_node(
            cfg=chat_agent.cfg,
            tracer=tracer,
            trace_full_state=trace_full_state,
        ),
    )
    builder.add_node(
        "tool_vectorize",
        build_tool_vectorize_node(
            cfg=chat_agent.cfg,
            tracer=tracer,
            trace_full_state=trace_full_state,
        ),
    )
    builder.add_node(
        "tool_push_queue_message",
        build_tool_push_queue_message_node(
            cfg=chat_agent.cfg,
            chat_agent=chat_agent,
            tracer=tracer,
            trace_full_state=trace_full_state,
        ),
    )
    builder.add_node("respond_fast", respond_fast_node)
    builder.add_node("respond_deep", respond_deep_node)
    builder.add_edge(START, "human_input")
    builder.add_edge("human_input", "recall_memories")
    builder.add_edge("recall_memories", "plan")
    builder.add_edge("plan", "dispatch")
    builder.add_edge("tool_list_sessions", "dispatch")
    builder.add_edge("tool_analyze_summary", "dispatch")
    builder.add_edge("tool_query_conversation", "dispatch")
    builder.add_edge("tool_save_to_siyuan", "dispatch")
    builder.add_edge("tool_memorize", "dispatch")
    builder.add_edge("tool_recall_memory", "dispatch")
    builder.add_edge("tool_search_knowledge", "dispatch")
    builder.add_edge("tool_vectorize", "dispatch")
    builder.add_edge("tool_push_queue_message", "dispatch")
    builder.add_edge("respond_fast", "dispatch")
    builder.add_edge("respond_deep", "dispatch")
    builder.add_conditional_edges(
        "extract_session_id",
        _next_node_after_extract_session_id,
        {
            "tool_analyze_summary": "tool_analyze_summary",
            "tool_query_conversation": "tool_query_conversation",
        },
    )
    builder.add_conditional_edges(
        "dispatch",
        _next_node_from_dispatch,
        {
            "tool_list_sessions": "tool_list_sessions",
            "extract_session_id": "extract_session_id",
            "tool_save_to_siyuan": "tool_save_to_siyuan",
            "tool_memorize": "tool_memorize",
            "tool_recall_memory": "tool_recall_memory",
            "tool_search_knowledge": "tool_search_knowledge",
            "tool_vectorize": "tool_vectorize",
            "tool_push_queue_message": "tool_push_queue_message",
            "respond_fast": "respond_fast",
            "respond_deep": "respond_deep",
            "__end__": END,
        },
    )
    return builder.compile()


class LangGraphChatSession:
    """Owns the LangGraph app and in-memory chat state for one REPL session."""

    def __init__(
        self,
        cfg: AgentConfig,
        emit: Callable[[str], None],
        on_thinking: Callable[[], None] | None = None,
        trace_full_state: bool = False,
    ) -> None:
        self.cfg = cfg
        self._emit = emit
        self._trace_full_state = trace_full_state
        self._chat_agent = LangGraphRouterChatAgent(cfg=cfg, on_thinking=on_thinking)
        self._tracer = build_phoenix_tracer(cfg)
        self._graph = None
        self._state: LangGraphChatState = new_langgraph_chat_state()

    @classmethod
    async def create(
        cls,
        cfg: AgentConfig,
        emit: Callable[[str], None],
        on_thinking: Callable[[], None] | None = None,
        trace_full_state: bool = False,
    ) -> "LangGraphChatSession":
        session = cls(
            cfg=cfg,
            emit=emit,
            on_thinking=on_thinking,
            trace_full_state=trace_full_state,
        )
        session._graph = await build_langgraph_chat_graph(
            session._chat_agent,
            session._tracer,
            trace_full_state=trace_full_state,
        )
        return session

    async def reset(self) -> None:
        self._state = new_langgraph_chat_state()

    async def handle_user_input(self, text: str) -> str:
        if self._graph is None:
            raise RuntimeError("LangGraph chat application is not initialized.")
        if self._tracer is None:
            payload = set_state_fields(dict(self._state), incoming_prompt=text)
            result = await self._graph.ainvoke(payload)
            state = set_state_fields(ensure_langgraph_state(result), incoming_prompt="")
            self._state = state
            return normalize_output(get_state_field(self._state, "latest_assistant_message"))
        with self._tracer.start_as_current_span("langgraph-chat-turn") as span:
            if self._trace_full_state:
                _trace_full_state_snapshot(span, self._state, label="before")
            payload = set_state_fields(dict(self._state), incoming_prompt=text)
            span.set_attribute("framework", "langgraph")
            span.set_attribute("agent.name", self.cfg.agent_name)
            span.set_attribute("input.value", text)
            span.set_attribute("input.turn_count", int(get_state_field(self._state, "turn_count")))
            result = await self._graph.ainvoke(payload)
            state = set_state_fields(ensure_langgraph_state(result), incoming_prompt="")
            self._state = state
            output = normalize_output(get_state_field(self._state, "latest_assistant_message"))
            span.set_attribute(
                "route.kind", normalize_output(get_state_field(self._state, "route_kind"))
            )
            span.set_attribute(
                "route.model", normalize_output(get_state_field(self._state, "route_model"))
            )
            span.set_attribute(
                "llm.model_name", normalize_output(get_state_field(self._state, "reply_model"))
            )
            if normalize_output(get_state_field(self._state, "tool_name")):
                span.set_attribute(
                    "tool.name", normalize_output(get_state_field(self._state, "tool_name"))
                )
            if normalize_output(get_state_field(self._state, "latest_session_id")):
                span.set_attribute(
                    "session.id",
                    normalize_output(get_state_field(self._state, "latest_session_id")),
                )
            span.set_attribute("output.value", output)
            if self._trace_full_state:
                _trace_full_state_snapshot(span, self._state, label="after")
            return output


async def run_langgraph_chat(
    cfg: AgentConfig,
    emit: Callable[[str], None] = print,
    on_thinking: Callable[[], None] | None = None,
    trace_full_state: bool = False,
) -> None:
    """Run the minimal LangGraph-backed chat REPL."""
    session = await LangGraphChatSession.create(
        cfg=cfg,
        emit=emit,
        on_thinking=on_thinking,
        trace_full_state=trace_full_state,
    )
    prompt_session = PromptSession()
    while True:
        try:
            raw = await prompt_session.prompt_async("You: ")
        except (EOFError, KeyboardInterrupt):
            return
        text = raw.strip()
        if not text:
            continue
        if text.lower() in {"/exit", "/quit"}:
            return
        if text.lower() == "/reset":
            await session.reset()
            continue
        if text.startswith("/"):
            emit("LangGraph chat supports only /reset, /exit, and /quit.")
            continue
        emit(await session.handle_user_input(text))
