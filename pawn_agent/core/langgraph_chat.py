"""Minimal LangGraph-managed chat path for evaluating dynamic context orchestration."""

from __future__ import annotations

from typing import Any, Callable, TypedDict

from prompt_toolkit import PromptSession

from pawn_agent.core.burr_chat import (
    _normalize_output,
    apply_assistant_message,
    apply_user_message,
)
from pawn_agent.core.langgraph_runtime import (
    PlainSmolagentsChatAgent,
    build_phoenix_tracer,
)
from pawn_agent.utils.config import AgentConfig


class LangGraphChatState(TypedDict, total=False):
    """State carried through the minimal LangGraph chatbot."""

    incoming_prompt: str
    chat_history: list[dict[str, str]]
    latest_user_message: str
    latest_assistant_message: str
    turn_count: int


def new_langgraph_chat_state() -> LangGraphChatState:
    """Return the default LangGraph chat state."""
    return {
        "incoming_prompt": "",
        "chat_history": [],
        "latest_user_message": "",
        "latest_assistant_message": "",
        "turn_count": 0,
    }


def _import_langgraph_core():
    try:
        from langgraph.graph import END, START, StateGraph
    except ImportError as exc:  # pragma: no cover - exercised via CLI/runtime integration
        raise RuntimeError(
            "LangGraph chat requires the optional 'langgraph' dependency to be installed."
        ) from exc
    return StateGraph, START, END


async def build_langgraph_chat_graph(chat_agent: PlainSmolagentsChatAgent, tracer=None):
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

    async def ai_response_node(state: LangGraphChatState) -> LangGraphChatState:
        if tracer is None:
            latest_user_message = _normalize_output(state.get("latest_user_message", ""))
            reply = await chat_agent.reply(
                latest_user_message,
                list(state.get("chat_history", [])),
            )
            updated = apply_assistant_message(dict(state), reply)
            updated["incoming_prompt"] = ""
            return updated
        with tracer.start_as_current_span("langgraph-ai-response") as span:
            latest_user_message = _normalize_output(state.get("latest_user_message", ""))
            span.set_attribute("llm.model_name", chat_agent.model_name)
            span.set_attribute("input.value", latest_user_message)
            reply = await chat_agent.reply(
                latest_user_message,
                list(state.get("chat_history", [])),
            )
            updated = apply_assistant_message(dict(state), reply)
            updated["incoming_prompt"] = ""
            span.set_attribute("output.value", reply)
            return updated

    builder = StateGraph(LangGraphChatState)
    builder.add_node("human_input", human_input_node)
    builder.add_node("ai_response", ai_response_node)
    builder.add_edge(START, "human_input")
    builder.add_edge("human_input", "ai_response")
    builder.add_edge("ai_response", END)
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
        self._chat_agent = PlainSmolagentsChatAgent(cfg=cfg, on_thinking=on_thinking)
        self._tracer = build_phoenix_tracer(cfg)
        self._graph: Any | None = None
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

    def _build_turn_payload(self, text: str) -> LangGraphChatState:
        payload = dict(self._state)
        payload["incoming_prompt"] = text
        return payload

    def _apply_graph_result(self, result: object) -> str:
        state: LangGraphChatState = dict(result or ())
        state["incoming_prompt"] = ""
        self._state = state
        return _normalize_output(self._state["latest_assistant_message"])

    async def _run_turn(self, text: str) -> str:
        assert self._graph is not None
        result = await self._graph.ainvoke(self._build_turn_payload(text))
        return self._apply_graph_result(result)

    async def handle_user_input(self, text: str) -> str:
        if self._graph is None:
            raise RuntimeError("LangGraph chat application is not initialized.")
        if self._tracer is None:
            return await self._run_turn(text)
        with self._tracer.start_as_current_span("langgraph-chat-turn") as span:
            span.set_attribute("framework", "langgraph")
            span.set_attribute("agent.name", self.cfg.agent_name)
            span.set_attribute("input.value", text)
            span.set_attribute("input.turn_count", int(self._state.get("turn_count", 0)))
            output = await self._run_turn(text)
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
