from __future__ import annotations

import asyncio
import sys
from datetime import datetime
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

from pawn_agent.core.langgraph_chat import (
    LangGraphChatSession,
    LangGraphRouterChatAgent,
    build_langgraph_chat_graph,
    build_phoenix_tracer,
    new_langgraph_chat_state,
)
from pawn_agent.tools.list_sessions import build as build_list_sessions_tool
from pawn_agent.tools.list_sessions import list_sessions_impl
from pawn_agent.utils.config import load_config


def test_new_langgraph_chat_state_resets_history() -> None:
    assert new_langgraph_chat_state() == {
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


def test_load_config_parses_langgraph_fast_model(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "agent:\n"
        "  openai:\n"
        "    fast_model: gemma4:e4b\n"
        "    model: gemma4:26b\n"
        "    base_url: http://localhost:11434/v1\n"
        "    api_key: ollama\n",
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_file))

    assert cfg.langgraph_fast_model == "openai:gemma4:e4b"
    assert cfg.langgraph_deep_model == "openai:gemma4:26b"
    assert cfg.langgraph_base_url == "http://localhost:11434/v1"
    assert cfg.langgraph_api_key == "ollama"


def test_build_langgraph_chat_graph_registers_nodes_and_edges() -> None:
    calls: dict[str, object] = {}

    class FakeCompiledGraph:
        async def ainvoke(self, payload):
            return payload

    class FakeStateGraph:
        def __init__(self, state_schema) -> None:
            calls["state_schema"] = state_schema
            calls["nodes"] = {}
            calls["edges"] = []

        def add_node(self, name, fn) -> None:
            calls["nodes"][name] = fn

        def add_edge(self, start, end) -> None:
            calls["edges"].append((start, end))

        def add_conditional_edges(self, start, fn, path_map) -> None:
            calls["conditional"] = (start, fn, path_map)

        def compile(self):
            calls["compiled"] = True
            return FakeCompiledGraph()

    fake_agent = SimpleNamespace(
        last_route_kind="reply_fast",
        last_route_model="openai:gemma4:e4b",
        last_reply_model="openai:gemma4:e4b",
        last_tool_name="",
    )
    with patch(
        "pawn_agent.core.langgraph_chat._import_langgraph_core",
        return_value=(FakeStateGraph, "__start__", "__end__"),
    ):
        compiled = asyncio.run(build_langgraph_chat_graph(chat_agent=fake_agent))

    assert isinstance(compiled, FakeCompiledGraph)
    assert set(calls["nodes"]) == {
        "human_input",
        "route",
        "tool_list_sessions",
        "respond_fast",
        "respond_deep",
    }
    assert calls["edges"] == [
        ("__start__", "human_input"),
        ("human_input", "route"),
        ("tool_list_sessions", "respond_fast"),
        ("respond_fast", "__end__"),
        ("respond_deep", "__end__"),
    ]
    assert calls["conditional"][0] == "route"
    assert calls["conditional"][2] == {
        "tool_list_sessions": "tool_list_sessions",
        "respond_fast": "respond_fast",
        "respond_deep": "respond_deep",
    }
    assert calls["compiled"] is True


def test_load_config_parses_phoenix_section(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "phoenix:\n"
        "  enabled: true\n"
        "  endpoint: http://localhost:6006/v1/traces\n"
        "  project_name: parakeet-evals\n"
        "  protocol: grpc\n"
        "  api_key: secret-token\n",
        encoding="utf-8",
    )

    cfg = load_config(str(cfg_file))

    assert cfg.phoenix_enabled is True
    assert cfg.phoenix_endpoint == "http://localhost:6006/v1/traces"
    assert cfg.phoenix_project_name == "parakeet-evals"
    assert cfg.phoenix_protocol == "grpc"
    assert cfg.phoenix_api_key == "secret-token"


def test_build_phoenix_tracer_is_disabled_by_default(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text("agent:\n  openai:\n    model: gpt-4o\n", encoding="utf-8")

    cfg = load_config(str(cfg_file))

    assert build_phoenix_tracer(cfg) is None


def test_build_phoenix_tracer_passes_configuration() -> None:
    tracer = object()
    register_calls: dict[str, object] = {}
    trace_calls: list[str] = []

    class FakeTraceApi:
        @staticmethod
        def get_tracer(name: str):
            trace_calls.append(name)
            return tracer

    cfg = SimpleNamespace(
        phoenix_enabled=True,
        phoenix_endpoint="http://localhost:6006/v1/traces",
        phoenix_project_name="parakeet-langgraph",
        phoenix_protocol="http/protobuf",
        phoenix_api_key="secret-token",
    )

    def fake_register(**kwargs):
        register_calls.update(kwargs)

    with (
        patch("pawn_agent.core.langgraph_chat._import_phoenix_register", return_value=fake_register),
        patch("pawn_agent.core.langgraph_chat._import_trace_api", return_value=FakeTraceApi),
    ):
        built = build_phoenix_tracer(cfg)

    assert built is tracer
    assert register_calls == {
        "project_name": "parakeet-langgraph",
        "endpoint": "http://localhost:6006/v1/traces",
        "protocol": "http/protobuf",
        "auto_instrument": False,
        "batch": True,
        "headers": {"authorization": "Bearer secret-token"},
    }
    assert trace_calls == ["pawn_agent.core.langgraph_chat"]


def test_list_sessions_impl_formats_rows() -> None:
    cfg = SimpleNamespace(db_dsn="postgresql://dummy")
    created_at = datetime(2026, 4, 19, 15, 30)
    rows = [
        SimpleNamespace(
            session_id="sess-123",
            segments=8,
            first_start=0.0,
            last_end=95.0,
            last_updated=created_at,
        )
    ]

    class FakeResult:
        def all(self):
            return rows

    class FakeDbSession:
        def __init__(self, _engine) -> None:
            pass

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def execute(self, _stmt):
            return FakeResult()

    class FakeFuncs:
        @staticmethod
        def count(_value):
            return SimpleNamespace(label=lambda _name: "count")

        @staticmethod
        def min(_value):
            return SimpleNamespace(label=lambda _name: "min")

        @staticmethod
        def max(_value):
            return SimpleNamespace(label=lambda _name: "max", desc=lambda: "desc")

    class FakeStmt:
        def group_by(self, *_args):
            return self

        def order_by(self, *_args):
            return self

        def where(self, *_args):
            return self

        def limit(self, _value):
            return self

    fake_segment = SimpleNamespace(
        session_id=SimpleNamespace(ilike=lambda _value: "ilike"),
        id="id",
        start_time="start_time",
        end_time="end_time",
        created_at="created_at",
    )

    fake_sqlalchemy = ModuleType("sqlalchemy")
    fake_sqlalchemy.create_engine = lambda _dsn: object()
    fake_sqlalchemy.func = FakeFuncs
    fake_sqlalchemy.select = lambda *_args: FakeStmt()

    fake_sqlalchemy_orm = ModuleType("sqlalchemy.orm")
    fake_sqlalchemy_orm.Session = FakeDbSession

    fake_db = ModuleType("pawn_agent.utils.db")
    fake_db.TranscriptionSegment = fake_segment

    with patch.dict(
        sys.modules,
        {
            "sqlalchemy": fake_sqlalchemy,
            "sqlalchemy.orm": fake_sqlalchemy_orm,
            "pawn_agent.utils.db": fake_db,
        },
    ):
        result = list_sessions_impl(cfg, limit=5)

    assert "Found 1 session(s):" in result
    assert "sess-123" in result
    assert "segments: 8" in result
    assert "duration: 1m 35s" in result
    assert "updated: 2026-04-19 15:30" in result


def test_list_sessions_build_delegates_to_shared_helper() -> None:
    cfg = SimpleNamespace()
    captured: dict[str, object] = {}

    class FakeTool:
        def __init__(self, fn) -> None:
            captured["fn"] = fn

    fake_pydantic_ai = ModuleType("pydantic_ai")
    fake_pydantic_ai.Tool = FakeTool

    with (
        patch.dict(sys.modules, {"pydantic_ai": fake_pydantic_ai}),
        patch(
            "pawn_agent.tools.list_sessions.list_sessions_impl",
            return_value="Found 2 session(s):",
        ) as mock_impl,
    ):
        tool_obj = build_list_sessions_tool(cfg)
        reply = captured["fn"](name_filter="demo", limit=3)

    assert isinstance(tool_obj, FakeTool)
    assert reply == "Found 2 session(s):"
    mock_impl.assert_called_once_with(cfg, name_filter="demo", limit=3)


def test_langgraph_router_agent_routes_fast_reply(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "agent:\n"
        "  openai:\n"
        "    fast_model: gemma4:e4b\n"
        "    model: gemma4:26b\n"
        "    base_url: http://localhost:11434/v1\n"
        "    api_key: ollama\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))
    calls: list[tuple[str, str, object]] = []

    class FakeAgent:
        def __init__(self, cfg, on_thinking=None) -> None:
            self.cfg = cfg

        async def reply(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
            calls.append((self.cfg.pydantic_model, user_prompt, list(chat_history)))
            if user_prompt.startswith("You are a routing model"):
                return "reply_fast"
            return "fast:hello"

    with patch("pawn_agent.core.langgraph_chat.PlainPydanticChatAgent", FakeAgent):
        agent = LangGraphRouterChatAgent(cfg)
        route = asyncio.run(agent.route("hello", [{"role": "user", "content": "hello"}]))
        reply = asyncio.run(
            agent.respond_fast("hello", [{"role": "user", "content": "hello"}])
        )

    assert route == "reply_fast"
    assert reply == "fast:hello"
    assert calls[0][0] == "openai:gemma4:e4b"
    assert calls[1][0] == "openai:gemma4:e4b"
    assert calls[1][1] == "hello"
    assert agent.last_route_kind == "reply_fast"
    assert agent.last_route_model == "openai:gemma4:e4b"
    assert agent.last_reply_model == "openai:gemma4:e4b"
    assert agent.last_tool_name == ""


def test_langgraph_router_agent_routes_deep_reply(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "agent:\n"
        "  openai:\n"
        "    fast_model: gemma4:e4b\n"
        "    model: gemma4:26b\n"
        "    base_url: http://localhost:11434/v1\n"
        "    api_key: ollama\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))
    calls: list[tuple[str, str]] = []

    class FakeAgent:
        def __init__(self, cfg, on_thinking=None) -> None:
            self.cfg = cfg

        async def reply(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
            calls.append((self.cfg.pydantic_model, user_prompt))
            if user_prompt.startswith("You are a routing model"):
                return "reply_deep"
            return "deep:analysis"

    with patch("pawn_agent.core.langgraph_chat.PlainPydanticChatAgent", FakeAgent):
        agent = LangGraphRouterChatAgent(cfg)
        route = asyncio.run(agent.route("analyze this deeply", []))
        reply = asyncio.run(agent.respond_deep("analyze this deeply", []))

    assert route == "reply_deep"
    assert reply == "deep:analysis"
    assert calls[0][0] == "openai:gemma4:e4b"
    assert calls[1][0] == "openai:gemma4:26b"
    assert agent.last_route_kind == "reply_deep"
    assert agent.last_reply_model == "openai:gemma4:26b"
    assert agent.last_tool_name == ""


def test_langgraph_router_agent_routes_to_list_sessions_tool(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "agent:\n"
        "  openai:\n"
        "    fast_model: gemma4:e4b\n"
        "    model: gemma4:26b\n"
        "    base_url: http://localhost:11434/v1\n"
        "    api_key: ollama\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))
    calls: list[tuple[str, str]] = []

    class FakeAgent:
        def __init__(self, cfg, on_thinking=None) -> None:
            self.cfg = cfg

        async def reply(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
            calls.append((self.cfg.pydantic_model, user_prompt))
            if user_prompt.startswith("You are a routing model"):
                return "tool_list_sessions"
            return "Here are the latest sessions."

    with (
        patch("pawn_agent.core.langgraph_chat.PlainPydanticChatAgent", FakeAgent),
        patch(
            "pawn_agent.core.langgraph_chat.list_sessions_impl",
            return_value="Found 1 session(s):\n  sess-123",
        ) as mock_tool,
    ):
        agent = LangGraphRouterChatAgent(cfg)
        route = asyncio.run(agent.route("show me the latest sessions", []))
        tool_output = agent.run_list_sessions_tool()
        reply = asyncio.run(
            agent.respond_fast(
                "show me the latest sessions",
                [],
                tool_output=tool_output,
            )
        )

    assert route == "tool_list_sessions"
    assert reply == "Here are the latest sessions."
    assert calls[0][0] == "openai:gemma4:e4b"
    assert calls[1][0] == "openai:gemma4:e4b"
    assert "Tool result:\nFound 1 session(s):\n  sess-123" in calls[1][1]
    mock_tool.assert_called_once_with(cfg)
    assert agent.last_route_kind == "tool_list_sessions"
    assert agent.last_reply_model == "openai:gemma4:e4b"
    assert agent.last_tool_name == "list_sessions"


def test_langgraph_chat_session_handles_turns_and_reset(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "agent:\n"
        "  openai:\n"
        "    fast_model: gemma4:e4b\n"
        "    model: gemma4:26b\n"
        "    base_url: http://localhost:11434/v1\n"
        "    api_key: ollama\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))

    payloads: list[dict[str, object]] = []

    class FakeGraph:
        async def ainvoke(self, payload):
            payloads.append(dict(payload))
            history = list(payload.get("chat_history", []))
            history.append({"role": "user", "content": payload["incoming_prompt"]})
            history.append({"role": "assistant", "content": f"echo:{payload['incoming_prompt']}"})
            return {
                "incoming_prompt": "",
                "chat_history": history,
                "latest_user_message": payload["incoming_prompt"],
                "latest_assistant_message": f"echo:{payload['incoming_prompt']}",
                "turn_count": int(payload.get("turn_count", 0)) + 1,
            }

    with (
        patch("pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=object()),
        patch("pawn_agent.core.langgraph_chat.build_langgraph_chat_graph", return_value=FakeGraph()),
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        first = asyncio.run(session.handle_user_input("hello"))
        second = asyncio.run(session.handle_user_input("again"))
        asyncio.run(session.reset())

    assert first == "echo:hello"
    assert second == "echo:again"
    assert payloads[0]["chat_history"] == []
    assert payloads[1]["chat_history"] == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "echo:hello"},
    ]
    assert session._state == new_langgraph_chat_state()


def test_langgraph_chat_session_emits_phoenix_spans(tmp_path: Path) -> None:
    cfg_file = tmp_path / "pawnai.yaml"
    cfg_file.write_text(
        "agent:\n"
        "  name: Bob\n"
        "  openai:\n"
        "    fast_model: gemma4:e4b\n"
        "    model: gemma4:26b\n"
        "    base_url: http://localhost:11434/v1\n"
        "    api_key: ollama\n",
        encoding="utf-8",
    )
    cfg = load_config(str(cfg_file))
    span_events: list[tuple[str, str, object | None, object | None]] = []

    class FakeSpan:
        def __init__(self, name: str) -> None:
            self._name = name

        def __enter__(self):
            span_events.append(("enter", self._name, None, None))
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            span_events.append(("exit", self._name, None, None))

        def set_attribute(self, key: str, value: object) -> None:
            span_events.append(("attr", self._name, key, value))

    class FakeTracer:
        def start_as_current_span(self, name: str):
            return FakeSpan(name)

    class FakeCompiledGraph:
        def __init__(self, nodes) -> None:
            self._nodes = nodes
            self._route_fn = None

        def set_route_fn(self, route_fn) -> None:
            self._route_fn = route_fn

        async def ainvoke(self, payload):
            state = dict(payload)
            state = self._nodes["human_input"](state)
            state = await self._nodes["route"](state)
            next_node = self._route_fn(state)
            if next_node == "tool_list_sessions":
                state = self._nodes["tool_list_sessions"](state)
                next_node = "respond_fast"
            return await self._nodes[next_node](state)

    class FakeStateGraph:
        def __init__(self, _state_schema) -> None:
            self._nodes: dict[str, object] = {}
            self._route_fn = None

        def add_node(self, name, fn) -> None:
            self._nodes[name] = fn

        def add_edge(self, _start, _end) -> None:
            return None

        def add_conditional_edges(self, _start, fn, _path_map) -> None:
            self._route_fn = fn

        def compile(self):
            graph = FakeCompiledGraph(self._nodes)
            graph.set_route_fn(self._route_fn)
            return graph

    class FakeChatAgent:
        last_route_kind = ""
        last_route_model = "openai:gemma4:e4b"
        last_reply_model = ""
        last_tool_name = ""

        async def route(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
            assert user_prompt == "show latest sessions"
            assert chat_history == [{"role": "user", "content": "show latest sessions"}]
            self.last_route_kind = "tool_list_sessions"
            self.last_tool_name = ""
            return "tool_list_sessions"

        def run_list_sessions_tool(self) -> str:
            self.last_tool_name = "list_sessions"
            return "Found 1 session(s):\n  sess-123"

        async def respond_fast(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            assert user_prompt == "show latest sessions"
            assert chat_history == [{"role": "user", "content": "show latest sessions"}]
            assert tool_output == "Found 1 session(s):\n  sess-123"
            self.last_reply_model = "openai:gemma4:e4b"
            return "Here are the latest sessions."

        async def respond_deep(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
            raise AssertionError("Deep responder should not run in tool path test.")

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch("pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=FakeChatAgent()),
        patch("pawn_agent.core.langgraph_chat.build_phoenix_tracer", return_value=FakeTracer()),
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        reply = asyncio.run(session.handle_user_input("show latest sessions"))

    assert reply == "Here are the latest sessions."
    span_names = [event[1] for event in span_events if event[0] == "enter"]
    assert span_names == [
        "langgraph-chat-turn",
        "langgraph-human-input",
        "langgraph-route",
        "langgraph-tool-list-sessions",
        "langgraph-respond-fast",
    ]
    assert ("attr", "langgraph-route", "output.route_kind", "tool_list_sessions") in span_events
    assert ("attr", "langgraph-tool-list-sessions", "tool.name", "list_sessions") in span_events
    assert ("attr", "langgraph-respond-fast", "llm.model_name", "openai:gemma4:e4b") in span_events
    assert ("attr", "langgraph-respond-fast", "tool.name", "list_sessions") in span_events
