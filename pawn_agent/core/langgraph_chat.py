"""Minimal LangGraph-managed chat path for evaluating dynamic context orchestration."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable, TypedDict

from prompt_toolkit import PromptSession

from pawn_agent.core.burr_chat import (
    PlainPydanticChatAgent,
    _normalize_output,
    apply_assistant_message,
    apply_user_message,
)
from pawn_agent.tools.list_sessions import list_sessions_impl
from pawn_agent.utils.config import AgentConfig


class LangGraphChatState(TypedDict, total=False):
    """State carried through the minimal LangGraph chatbot."""

    incoming_prompt: str
    chat_history: list[dict[str, str]]
    latest_user_message: str
    latest_assistant_message: str
    turn_count: int
    route_kind: str
    route_model: str
    reply_model: str
    tool_name: str
    tool_output: str


def new_langgraph_chat_state() -> LangGraphChatState:
    """Return the default LangGraph chat state."""
    return {
        "incoming_prompt": "",
        "chat_history": [],
        "latest_user_message": "",
        "latest_assistant_message": "",
        "turn_count": 0,
        "route_kind": "",
        "route_model": "",
        "reply_model": "",
        "tool_name": "",
        "tool_output": "",
    }


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


class LangGraphRouterChatAgent:
    """Forward-only fast/deep router used by the LangGraph evaluation path."""

    VALID_ROUTES = {"reply_fast", "reply_deep", "tool_list_sessions"}

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
            role = _normalize_output(item.get("role", "")).strip().lower()
            content = _normalize_output(item.get("content", "")).strip()
            if not content:
                continue
            speaker = "User" if role == "user" else "Assistant"
            lines.append(f"{speaker}: {content}")
        return "\n".join(lines) if lines else "(no prior conversation)"

    def _router_prompt(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
        return (
            "You are a routing model for a forward-only LangGraph chat system.\n"
            "Choose exactly one label and return only that label.\n"
            "Labels:\n"
            "- reply_fast: simple conversational reply, lightweight reasoning, or normal follow-up\n"
            "- reply_deep: analysis, synthesis, complex reasoning, or anything needing deeper thought\n"
            "- tool_list_sessions: the user asks for recent/latest/available sessions, or needs session discovery before analysis\n\n"
            "Conversation so far:\n"
            f"{self._history_transcript(chat_history, user_prompt)}\n\n"
            "Current user message:\n"
            f"{user_prompt}\n"
        )

    def _normalize_route(self, route_reply: str) -> str:
        route = (
            _normalize_output(route_reply).strip().lower().splitlines()[0].strip()
            if route_reply
            else ""
        )
        if route in self.VALID_ROUTES:
            return route
        for candidate in self.VALID_ROUTES:
            if candidate in route:
                return candidate
        return "reply_deep"

    async def route(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
        if self._on_thinking is not None:
            self._on_thinking()

        self.last_route_model = self.cfg.langgraph_fast_model
        self.last_reply_model = ""
        self.last_tool_name = ""
        self.model_name = self.cfg.langgraph_fast_model
        route_reply = await self._fast_agent.reply(self._router_prompt(user_prompt, chat_history), [])
        self.last_route_kind = self._normalize_route(route_reply)
        return self.last_route_kind

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

    async def respond_deep(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
        self.last_reply_model = self.cfg.langgraph_deep_model
        self.model_name = self.last_reply_model
        return await self._deep_agent.reply(user_prompt, chat_history)

    def run_list_sessions_tool(self) -> str:
        self.last_tool_name = "list_sessions"
        return list_sessions_impl(self.cfg)


def _next_node_from_route(state: LangGraphChatState) -> str:
    route_kind = _normalize_output(state.get("route_kind", "")).strip()
    if route_kind == "tool_list_sessions":
        return "tool_list_sessions"
    if route_kind == "reply_fast":
        return "respond_fast"
    return "respond_deep"


async def build_langgraph_chat_graph(chat_agent: LangGraphRouterChatAgent, tracer=None):
    """Build the minimal LangGraph chatbot graph."""
    StateGraph, START, END = _import_langgraph_core()

    def human_input_node(state: LangGraphChatState) -> LangGraphChatState:
        if tracer is None:
            prompt = _normalize_output(state.get("incoming_prompt", ""))
            updated = apply_user_message(dict(state), prompt)
            updated["incoming_prompt"] = prompt
            return updated
        with tracer.start_as_current_span("langgraph-human-input") as span:
            prompt = _normalize_output(state.get("incoming_prompt", ""))
            span.set_attribute("input.value", prompt)
            updated = apply_user_message(dict(state), prompt)
            updated["incoming_prompt"] = prompt
            span.set_attribute("output.turn_count", updated["turn_count"])
            return updated

    async def route_node(state: LangGraphChatState) -> LangGraphChatState:
        latest_user_message = _normalize_output(state.get("latest_user_message", ""))
        chat_history = list(state.get("chat_history", []))
        if tracer is None:
            route_kind = await chat_agent.route(latest_user_message, chat_history)
            updated = dict(state)
            updated["route_kind"] = route_kind
            updated["route_model"] = chat_agent.last_route_model
            updated["tool_name"] = ""
            updated["tool_output"] = ""
            return updated
        with tracer.start_as_current_span("langgraph-route") as span:
            route_kind = await chat_agent.route(latest_user_message, chat_history)
            updated = dict(state)
            updated["route_kind"] = route_kind
            updated["route_model"] = chat_agent.last_route_model
            updated["tool_name"] = ""
            updated["tool_output"] = ""
            span.set_attribute("llm.model_name", chat_agent.last_route_model)
            span.set_attribute("input.value", latest_user_message)
            span.set_attribute("output.route_kind", route_kind)
            return updated

    def tool_list_sessions_node(state: LangGraphChatState) -> LangGraphChatState:
        if tracer is None:
            tool_output = chat_agent.run_list_sessions_tool()
            updated = dict(state)
            updated["tool_name"] = chat_agent.last_tool_name
            updated["tool_output"] = tool_output
            return updated
        with tracer.start_as_current_span("langgraph-tool-list-sessions") as span:
            tool_output = chat_agent.run_list_sessions_tool()
            updated = dict(state)
            updated["tool_name"] = chat_agent.last_tool_name
            updated["tool_output"] = tool_output
            span.set_attribute("tool.name", chat_agent.last_tool_name)
            span.set_attribute("output.value", tool_output)
            return updated

    async def respond_fast_node(state: LangGraphChatState) -> LangGraphChatState:
        latest_user_message = _normalize_output(state.get("latest_user_message", ""))
        chat_history = list(state.get("chat_history", []))
        tool_output = _normalize_output(state.get("tool_output", ""))
        if tracer is None:
            reply = await chat_agent.respond_fast(
                latest_user_message,
                chat_history,
                tool_output=tool_output,
            )
            updated = apply_assistant_message(dict(state), reply)
            updated["incoming_prompt"] = ""
            updated["reply_model"] = chat_agent.last_reply_model
            return updated
        with tracer.start_as_current_span("langgraph-respond-fast") as span:
            reply = await chat_agent.respond_fast(
                latest_user_message,
                chat_history,
                tool_output=tool_output,
            )
            updated = apply_assistant_message(dict(state), reply)
            updated["incoming_prompt"] = ""
            updated["reply_model"] = chat_agent.last_reply_model
            span.set_attribute("llm.model_name", chat_agent.last_reply_model)
            span.set_attribute("input.value", latest_user_message)
            if _normalize_output(state.get("tool_name", "")):
                span.set_attribute("tool.name", _normalize_output(state.get("tool_name", "")))
            span.set_attribute("output.value", reply)
            return updated

    async def respond_deep_node(state: LangGraphChatState) -> LangGraphChatState:
        latest_user_message = _normalize_output(state.get("latest_user_message", ""))
        chat_history = list(state.get("chat_history", []))
        if tracer is None:
            reply = await chat_agent.respond_deep(latest_user_message, chat_history)
            updated = apply_assistant_message(dict(state), reply)
            updated["incoming_prompt"] = ""
            updated["reply_model"] = chat_agent.last_reply_model
            return updated
        with tracer.start_as_current_span("langgraph-respond-deep") as span:
            reply = await chat_agent.respond_deep(latest_user_message, chat_history)
            updated = apply_assistant_message(dict(state), reply)
            updated["incoming_prompt"] = ""
            updated["reply_model"] = chat_agent.last_reply_model
            span.set_attribute("llm.model_name", chat_agent.last_reply_model)
            span.set_attribute("input.value", latest_user_message)
            span.set_attribute("output.value", reply)
            return updated

    builder = StateGraph(LangGraphChatState)
    builder.add_node("human_input", human_input_node)
    builder.add_node("route", route_node)
    builder.add_node("tool_list_sessions", tool_list_sessions_node)
    builder.add_node("respond_fast", respond_fast_node)
    builder.add_node("respond_deep", respond_deep_node)
    builder.add_edge(START, "human_input")
    builder.add_edge("human_input", "route")
    builder.add_conditional_edges(
        "route",
        _next_node_from_route,
        {
            "tool_list_sessions": "tool_list_sessions",
            "respond_fast": "respond_fast",
            "respond_deep": "respond_deep",
        },
    )
    builder.add_edge("tool_list_sessions", "respond_fast")
    builder.add_edge("respond_fast", END)
    builder.add_edge("respond_deep", END)
    return builder.compile()


class LangGraphChatSession:
    """Owns the LangGraph app and in-memory chat state for one REPL session."""

    def __init__(
        self,
        cfg: AgentConfig,
        emit: Callable[[str], None],
        on_thinking: Callable[[], None] | None = None,
    ) -> None:
        self.cfg = cfg
        self._emit = emit
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
    ) -> "LangGraphChatSession":
        session = cls(cfg=cfg, emit=emit, on_thinking=on_thinking)
        session._graph = await build_langgraph_chat_graph(session._chat_agent, session._tracer)
        return session

    async def reset(self) -> None:
        self._state = new_langgraph_chat_state()

    async def handle_user_input(self, text: str) -> str:
        if self._graph is None:
            raise RuntimeError("LangGraph chat application is not initialized.")
        if self._tracer is None:
            payload = dict(self._state)
            payload["incoming_prompt"] = text
            result = await self._graph.ainvoke(payload)
            state = dict(result)
            state["incoming_prompt"] = ""
            self._state = state
            return _normalize_output(self._state["latest_assistant_message"])
        with self._tracer.start_as_current_span("langgraph-chat-turn") as span:
            payload = dict(self._state)
            payload["incoming_prompt"] = text
            span.set_attribute("framework", "langgraph")
            span.set_attribute("agent.name", self.cfg.agent_name)
            span.set_attribute("input.value", text)
            span.set_attribute("input.turn_count", int(self._state.get("turn_count", 0)))
            result = await self._graph.ainvoke(payload)
            state = dict(result)
            state["incoming_prompt"] = ""
            self._state = state
            output = _normalize_output(self._state["latest_assistant_message"])
            span.set_attribute("route.kind", _normalize_output(self._state.get("route_kind", "")))
            span.set_attribute("route.model", _normalize_output(self._state.get("route_model", "")))
            span.set_attribute("llm.model_name", _normalize_output(self._state.get("reply_model", "")))
            if _normalize_output(self._state.get("tool_name", "")):
                span.set_attribute("tool.name", _normalize_output(self._state.get("tool_name", "")))
            span.set_attribute("output.value", output)
            return output


async def run_langgraph_chat(
    cfg: AgentConfig,
    emit: Callable[[str], None] = print,
    on_thinking: Callable[[], None] | None = None,
) -> None:
    """Run the minimal LangGraph-backed chat REPL."""
    session = await LangGraphChatSession.create(cfg=cfg, emit=emit, on_thinking=on_thinking)
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
