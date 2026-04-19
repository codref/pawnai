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
from pawn_agent.core.langgraph_tools import (
    resolve_session_id_for_tool,
    resolve_session_id_from_list_output,
    run_list_sessions_tool,
    run_query_conversation_tool,
    run_save_to_siyuan_tool,
)
from pawn_agent.tools.list_sessions import build as build_list_sessions_tool
from pawn_agent.tools.list_sessions import list_sessions_impl
from pawn_agent.tools.query_conversation import build as build_query_conversation_tool
from pawn_agent.tools.query_conversation import query_conversation_impl
from pawn_agent.tools.save_to_siyuan import build as build_save_to_siyuan_tool
from pawn_agent.tools.save_to_siyuan import save_to_siyuan_impl
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
        "latest_session_id": "",
        "requested_session_id": "",
        "pending_save_to_siyuan": False,
        "latest_generated_content": "",
        "latest_generated_title": "",
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
            calls["conditionals"] = []

        def add_node(self, name, fn) -> None:
            calls["nodes"][name] = fn

        def add_edge(self, start, end) -> None:
            calls["edges"].append((start, end))

        def add_conditional_edges(self, start, fn, path_map) -> None:
            calls["conditionals"].append((start, fn, path_map))

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
        "extract_session_id",
        "tool_list_sessions",
        "tool_query_conversation",
        "tool_save_to_siyuan",
        "respond_fast",
        "respond_deep",
    }
    assert calls["edges"] == [
        ("__start__", "human_input"),
        ("human_input", "route"),
        ("extract_session_id", "tool_query_conversation"),
        ("tool_list_sessions", "respond_fast"),
        ("tool_query_conversation", "respond_fast"),
        ("tool_save_to_siyuan", "respond_fast"),
    ]
    assert calls["conditionals"][0][0] == "respond_fast"
    assert calls["conditionals"][0][2] == {
        "tool_save_to_siyuan": "tool_save_to_siyuan",
        "__end__": "__end__",
    }
    assert calls["conditionals"][1][0] == "respond_deep"
    assert calls["conditionals"][1][2] == {
        "tool_save_to_siyuan": "tool_save_to_siyuan",
        "__end__": "__end__",
    }
    assert calls["conditionals"][2][0] == "route"
    assert calls["conditionals"][2][2] == {
        "tool_list_sessions": "tool_list_sessions",
        "extract_session_id": "extract_session_id",
        "tool_save_to_siyuan": "tool_save_to_siyuan",
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


def test_query_conversation_impl_delegates_to_fetch_transcript() -> None:
    cfg = SimpleNamespace()

    with patch(
        "pawn_agent.tools.query_conversation.fetch_transcript",
        return_value="Speaker A: hi",
    ) as mock_fetch:
        reply = query_conversation_impl(cfg, "sess-123")

    assert reply == "Speaker A: hi"
    mock_fetch.assert_called_once_with(cfg, "sess-123")


def test_query_conversation_build_delegates_to_shared_helper() -> None:
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
            "pawn_agent.tools.query_conversation.query_conversation_impl",
            return_value="Speaker A: hi",
        ) as mock_impl,
    ):
        tool_obj = build_query_conversation_tool(cfg)
        reply = captured["fn"]("sess-123")

    assert isinstance(tool_obj, FakeTool)
    assert reply == "Speaker A: hi"
    mock_impl.assert_called_once_with(cfg, "sess-123")


def test_save_to_siyuan_impl_delegates_to_shared_helper() -> None:
    cfg = SimpleNamespace()

    with patch(
        "pawn_agent.tools.save_to_siyuan.do_save_to_siyuan",
        return_value="siyuan://blocks/doc-123",
    ) as mock_save:
        reply = save_to_siyuan_impl(
            cfg,
            session_id="sess-123",
            content="# Report\n\nHello",
            title="Report",
            path=None,
        )

    assert reply == "siyuan://blocks/doc-123"
    mock_save.assert_called_once_with(cfg, "sess-123", "Report", "# Report\n\nHello", None)


def test_save_to_siyuan_build_delegates_to_shared_helper() -> None:
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
            "pawn_agent.tools.save_to_siyuan.save_to_siyuan_impl",
            return_value="siyuan://blocks/doc-123",
        ) as mock_impl,
    ):
        tool_obj = build_save_to_siyuan_tool(cfg)
        reply = captured["fn"]("sess-123", "# Report\n\nHello", title="Report")

    assert isinstance(tool_obj, FakeTool)
    assert reply == "siyuan://blocks/doc-123"
    mock_impl.assert_called_once_with(
        cfg,
        session_id="sess-123",
        content="# Report\n\nHello",
        title="Report",
        path=None,
    )


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
            "pawn_agent.core.langgraph_tools.list_sessions_impl",
            return_value="Found 1 session(s):\n  sess-123",
        ) as mock_tool,
    ):
        agent = LangGraphRouterChatAgent(cfg)
        route = asyncio.run(agent.route("show me the latest sessions", []))
        tool_output = run_list_sessions_tool(cfg)
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


def test_langgraph_router_agent_routes_to_query_conversation_tool(tmp_path: Path) -> None:
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
                return "tool_query_conversation"
            if user_prompt.startswith("You extract a session id"):
                return "sess-123"
            return "Here is the transcript you asked for."

    with (
        patch("pawn_agent.core.langgraph_chat.PlainPydanticChatAgent", FakeAgent),
        patch(
            "pawn_agent.core.langgraph_tools.query_conversation_impl",
            return_value="Speaker A: hi\nSpeaker B: hello",
        ) as mock_tool,
    ):
        agent = LangGraphRouterChatAgent(cfg)
        route = asyncio.run(
            agent.route(
                "show session sess-123",
                [],
            )
        )
        session_id = asyncio.run(
            agent.extract_requested_session_id("show session sess-123", [])
        )
        tool_output = run_query_conversation_tool(cfg, session_id or "")
        reply = asyncio.run(
            agent.respond_fast(
                "show session sess-123",
                [],
                tool_output=tool_output,
            )
        )

    assert route == "tool_query_conversation"
    assert session_id == "sess-123"
    assert reply == "Here is the transcript you asked for."
    mock_tool.assert_called_once_with(cfg, "sess-123")
    assert "Tool result:\nSpeaker A: hi\nSpeaker B: hello" in calls[1][1]


def test_langgraph_router_agent_routes_to_save_to_siyuan_tool(tmp_path: Path) -> None:
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
                return "tool_save_to_siyuan"
            return "Saved the report to SiYuan."

    with (
        patch("pawn_agent.core.langgraph_chat.PlainPydanticChatAgent", FakeAgent),
        patch(
            "pawn_agent.core.langgraph_tools.save_to_siyuan_impl",
            return_value="siyuan://blocks/doc-123",
        ) as mock_tool,
    ):
        agent = LangGraphRouterChatAgent(cfg)
        route = asyncio.run(
            agent.route(
                "save the analysis into siyuan",
                [],
                latest_session_id="sess-123",
                latest_generated_content="# Report\n\nHello",
            )
        )
        tool_output = run_save_to_siyuan_tool(
            cfg,
            "sess-123",
            "# Report\n\nHello",
            title="Report",
        )
        reply = asyncio.run(
            agent.respond_fast(
                "save the analysis into siyuan",
                [],
                tool_output=tool_output,
            )
        )

    assert route == "tool_save_to_siyuan"
    assert reply == "Saved the report to SiYuan."
    mock_tool.assert_called_once_with(
        cfg,
        session_id="sess-123",
        content="# Report\n\nHello",
        title="Report",
        path=None,
    )
    assert "Latest generated content available:\nyes" in calls[0][1]
    assert "tool_save_to_siyuan" in calls[0][1]


def test_langgraph_session_tool_helper_prefers_current_turn_id() -> None:
    state = {
        "requested_session_id": "sess-456",
        "latest_session_id": "sess-123",
    }

    assert resolve_session_id_for_tool(state) == "sess-456"


def test_langgraph_session_tool_helper_falls_back_to_latest_session_id() -> None:
    state = {
        "requested_session_id": "",
        "latest_session_id": "sess-123",
    }

    assert resolve_session_id_for_tool(state) == "sess-123"


def test_langgraph_list_sessions_helper_prefers_matching_session() -> None:
    tool_output = (
        "Found 2 session(s):\n"
        "  sales-20260417  |  segments: 3  |  duration: 2m 10s  |  updated: 2026-04-17 11:00\n"
        "  oci-20260416  |  segments: 9  |  duration: 12m 00s  |  updated: 2026-04-16 15:57"
    )

    assert (
        resolve_session_id_from_list_output("retrieve last oci session", tool_output)
        == "oci-20260416"
    )


def test_langgraph_list_sessions_helper_falls_back_to_first_session() -> None:
    tool_output = (
        "Found 2 session(s):\n"
        "  sess-200  |  segments: 3  |  duration: 2m 10s  |  updated: 2026-04-17 11:00\n"
        "  sess-100  |  segments: 9  |  duration: 12m 00s  |  updated: 2026-04-16 15:57"
    )

    assert (
        resolve_session_id_from_list_output("retrieve the latest session", tool_output)
        == "sess-200"
    )


def test_langgraph_router_agent_session_id_extractor_returns_none_when_absent(tmp_path: Path) -> None:
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

    class FakeAgent:
        def __init__(self, cfg, on_thinking=None) -> None:
            self.cfg = cfg

        async def reply(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
            assert user_prompt.startswith("You extract a session id")
            return "NONE"

    with patch("pawn_agent.core.langgraph_chat.PlainPydanticChatAgent", FakeAgent):
        agent = LangGraphRouterChatAgent(cfg)
        session_id = asyncio.run(
            agent.extract_requested_session_id("extract action points", [])
        )

    assert session_id == ""


def test_langgraph_router_agent_session_id_extractor_returns_model_value(tmp_path: Path) -> None:
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

    class FakeAgent:
        def __init__(self, cfg, on_thinking=None) -> None:
            self.cfg = cfg

        async def reply(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
            assert user_prompt.startswith("You extract a session id")
            return "oci-20260416"

    with patch("pawn_agent.core.langgraph_chat.PlainPydanticChatAgent", FakeAgent):
        agent = LangGraphRouterChatAgent(cfg)
        session_id = asyncio.run(
            agent.extract_requested_session_id(
                "retrieve conversation (oci-20260416)",
                [],
            )
        )

    assert session_id == "oci-20260416"


def test_langgraph_router_prompt_guides_followup_report_to_conversation_tool(tmp_path: Path) -> None:
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

    with patch("pawn_agent.core.langgraph_chat.PlainPydanticChatAgent"):
        agent = LangGraphRouterChatAgent(cfg)
        prompt = agent._router_prompt(
            "give me an executive report",
            [{"role": "assistant", "content": "I retrieved session oci-20260416."}],
            latest_session_id="oci-20260416",
        )

    assert "executive report" in prompt
    assert "tool_query_conversation" in prompt
    assert "latest session is already in focus" in prompt
    assert "summary" in prompt
    assert "action points" in prompt
    assert "save it to siyuan in the same request" in prompt


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


def test_langgraph_chat_session_reuses_and_overrides_latest_session_id(tmp_path: Path) -> None:
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
            if next_node == "extract_session_id":
                state = await self._nodes[next_node](state)
                next_node = "tool_query_conversation"
            if next_node in {"tool_list_sessions", "tool_query_conversation"}:
                state = self._nodes[next_node](state)
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
        tool_session_ids: list[str] = []

        async def route(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            latest_session_id: str = "",
            latest_generated_content: str = "",
        ) -> str:
            if user_prompt == "extract action points":
                assert latest_session_id == "oci-20260416"
            self.last_route_kind = "tool_query_conversation"
            return "tool_query_conversation"

        async def extract_requested_session_id(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            latest_session_id: str = "",
        ) -> str:
            if user_prompt == "show session oci-20260416":
                return "oci-20260416"
            if user_prompt == "show session oci-20260417":
                return "oci-20260417"
            return ""

        async def respond_fast(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            self.last_reply_model = "openai:gemma4:e4b"
            return f"answer:{tool_output}"

        async def respond_deep(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
            raise AssertionError("Deep responder should not run in this test.")

    fake_chat_agent = FakeChatAgent()

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch("pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=fake_chat_agent),
        patch("pawn_agent.core.langgraph_chat.build_phoenix_tracer", return_value=None),
        patch(
            "pawn_agent.core.langgraph_tools.run_query_conversation_tool",
            side_effect=lambda cfg, session_id: (
                fake_chat_agent.tool_session_ids.append(session_id)
                or f"Transcript for {session_id}"
            ),
        ),
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        first = asyncio.run(session.handle_user_input("show session oci-20260416"))
        second = asyncio.run(session.handle_user_input("extract action points"))
        third = asyncio.run(session.handle_user_input("show session oci-20260417"))

    assert first == "answer:Transcript for oci-20260416"
    assert second == "answer:Transcript for oci-20260416"
    assert third == "answer:Transcript for oci-20260417"
    assert fake_chat_agent.tool_session_ids == [
        "oci-20260416",
        "oci-20260416",
        "oci-20260417",
    ]
    assert session._state["latest_session_id"] == "oci-20260417"
    assert session._state["requested_session_id"] == ""
    assert session._state["tool_output"] == ""


def test_langgraph_chat_session_promotes_latest_session_id_from_list_results(tmp_path: Path) -> None:
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
            if next_node == "extract_session_id":
                state = await self._nodes[next_node](state)
                next_node = "tool_query_conversation"
            if next_node in {"tool_list_sessions", "tool_query_conversation"}:
                state = self._nodes[next_node](state)
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

        def add_conditional_edges(self, start, fn, _path_map) -> None:
            if start == "route":
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

        async def route(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            latest_session_id: str = "",
            latest_generated_content: str = "",
        ) -> str:
            if user_prompt == "retrieve last oci session":
                assert latest_session_id == ""
                return "tool_list_sessions"
            if user_prompt == "extract an executive summary":
                assert latest_session_id == "oci-20260416"
                return "tool_query_conversation"
            raise AssertionError(f"Unexpected prompt: {user_prompt}")

        async def extract_requested_session_id(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            latest_session_id: str = "",
        ) -> str:
            assert user_prompt == "extract an executive summary"
            assert latest_session_id == "oci-20260416"
            return ""

        async def respond_fast(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            self.last_reply_model = "openai:gemma4:e4b"
            if user_prompt == "retrieve last oci session":
                return "The most recent session was oci-20260416."
            if user_prompt == "extract an executive summary":
                assert tool_output == "Transcript for oci-20260416"
                return "# Executive Summary\n\nFresh summary."
            raise AssertionError(f"Unexpected prompt: {user_prompt}")

        async def respond_deep(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
            raise AssertionError("Deep responder should not run in this test.")

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch(
            "pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent",
            return_value=FakeChatAgent(),
        ),
        patch("pawn_agent.core.langgraph_chat.build_phoenix_tracer", return_value=None),
        patch(
            "pawn_agent.core.langgraph_tools.run_list_sessions_tool",
            return_value=(
                "Found 2 session(s):\n"
                "  sales-20260417  |  segments: 3  |  duration: 2m 10s  |  updated: 2026-04-17 11:00\n"
                "  oci-20260416  |  segments: 9  |  duration: 12m 00s  |  updated: 2026-04-16 15:57"
            ),
        ),
        patch(
            "pawn_agent.core.langgraph_tools.run_query_conversation_tool",
            return_value="Transcript for oci-20260416",
        ),
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        first = asyncio.run(session.handle_user_input("retrieve last oci session"))
        second = asyncio.run(session.handle_user_input("extract an executive summary"))

    assert first == "The most recent session was oci-20260416."
    assert second == "# Executive Summary\n\nFresh summary."
    assert session._state["latest_session_id"] == "oci-20260416"


def test_langgraph_chat_session_saves_latest_generated_content_to_siyuan(tmp_path: Path) -> None:
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
            if next_node == "extract_session_id":
                state = await self._nodes[next_node](state)
                next_node = "tool_query_conversation"
            if next_node in {
                "tool_list_sessions",
                "tool_query_conversation",
                "tool_save_to_siyuan",
            }:
                state = self._nodes[next_node](state)
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

        async def route(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            latest_session_id: str = "",
            latest_generated_content: str = "",
        ) -> str:
            if user_prompt == "show session oci-20260416":
                return "tool_query_conversation"
            if user_prompt == "give me an executive report":
                assert latest_session_id == "oci-20260416"
                return "tool_query_conversation"
            if user_prompt == "save the analysis into siyuan":
                assert latest_session_id == "oci-20260416"
                assert latest_generated_content.startswith("# Executive Report")
                return "tool_save_to_siyuan"
            raise AssertionError(f"Unexpected prompt: {user_prompt}")

        async def extract_requested_session_id(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            latest_session_id: str = "",
        ) -> str:
            if user_prompt == "show session oci-20260416":
                return "oci-20260416"
            return ""

        async def respond_fast(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            self.last_reply_model = "openai:gemma4:e4b"
            if user_prompt == "show session oci-20260416":
                return "Loaded the session transcript."
            if user_prompt == "give me an executive report":
                return "# Executive Report\n\nThis is the report."
            if user_prompt == "save the analysis into siyuan":
                return f"Saved to SiYuan: {tool_output}"
            raise AssertionError(f"Unexpected prompt: {user_prompt}")

        async def respond_deep(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
            raise AssertionError("Deep responder should not run in this test.")

    fake_chat_agent = FakeChatAgent()

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch("pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=fake_chat_agent),
        patch("pawn_agent.core.langgraph_chat.build_phoenix_tracer", return_value=None),
        patch(
            "pawn_agent.core.langgraph_tools.run_query_conversation_tool",
            return_value="Speaker A: hi\nSpeaker B: hello",
        ),
        patch(
            "pawn_agent.core.langgraph_tools.run_save_to_siyuan_tool",
            return_value="siyuan://blocks/doc-123",
        ) as mock_save,
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        asyncio.run(session.handle_user_input("show session oci-20260416"))
        report = asyncio.run(session.handle_user_input("give me an executive report"))
        saved = asyncio.run(session.handle_user_input("save the analysis into siyuan"))

    assert report == "# Executive Report\n\nThis is the report."
    assert saved == "Saved to SiYuan: siyuan://blocks/doc-123"
    mock_save.assert_called_once_with(
        cfg,
        "oci-20260416",
        "# Executive Report\n\nThis is the report.",
        title="Executive Report",
    )
    assert session._state["latest_generated_content"] == "# Executive Report\n\nThis is the report."
    assert session._state["latest_generated_title"] == "Executive Report"
    assert session._state["latest_session_id"] == "oci-20260416"


def test_langgraph_chat_session_refreshes_session_analysis_before_saving(tmp_path: Path) -> None:
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
            if next_node == "extract_session_id":
                state = await self._nodes[next_node](state)
                next_node = "tool_query_conversation"
            if next_node in {
                "tool_list_sessions",
                "tool_query_conversation",
                "tool_save_to_siyuan",
            }:
                state = self._nodes[next_node](state)
                next_node = "respond_fast"
            state = await self._nodes[next_node](state)
            if state.get("pending_save_to_siyuan"):
                state = self._nodes["tool_save_to_siyuan"](state)
                state = await self._nodes["respond_fast"](state)
            return state

    class FakeStateGraph:
        def __init__(self, _state_schema) -> None:
            self._nodes: dict[str, object] = {}
            self._route_fn = None

        def add_node(self, name, fn) -> None:
            self._nodes[name] = fn

        def add_edge(self, _start, _end) -> None:
            return None

        def add_conditional_edges(self, start, fn, _path_map) -> None:
            if start == "route":
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

        async def route(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            latest_session_id: str = "",
            latest_generated_content: str = "",
        ) -> str:
            assert user_prompt == "extract an executive summary and save it to siyuan"
            assert latest_session_id == "oci-20260416"
            assert latest_generated_content == "# Old Summary\n\nStale"
            self.last_route_kind = "tool_query_conversation"
            return "tool_query_conversation"

        async def extract_requested_session_id(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            latest_session_id: str = "",
        ) -> str:
            assert latest_session_id == "oci-20260416"
            return ""

        async def respond_fast(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            self.last_reply_model = "openai:gemma4:e4b"
            if tool_output == "Speaker A: roadmap\nSpeaker B: deadlines":
                return "# Executive Summary\n\nFresh summary from transcript."
            if tool_output == "siyuan://blocks/doc-999":
                return f"Saved to SiYuan: {tool_output}"
            raise AssertionError(f"Unexpected tool output: {tool_output}")

        async def respond_deep(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
            raise AssertionError("Deep responder should not run in this test.")

    fake_chat_agent = FakeChatAgent()

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch("pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=fake_chat_agent),
        patch("pawn_agent.core.langgraph_chat.build_phoenix_tracer", return_value=None),
        patch(
            "pawn_agent.core.langgraph_tools.run_query_conversation_tool",
            return_value="Speaker A: roadmap\nSpeaker B: deadlines",
        ),
        patch(
            "pawn_agent.core.langgraph_tools.run_save_to_siyuan_tool",
            return_value="siyuan://blocks/doc-999",
        ) as mock_save,
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        session._state["latest_session_id"] = "oci-20260416"
        session._state["latest_generated_content"] = "# Old Summary\n\nStale"
        session._state["latest_generated_title"] = "Old Summary"
        reply = asyncio.run(
            session.handle_user_input("extract an executive summary and save it to siyuan")
        )

    assert reply == "Saved to SiYuan: siyuan://blocks/doc-999"
    mock_save.assert_called_once_with(
        cfg,
        "oci-20260416",
        "# Executive Summary\n\nFresh summary from transcript.",
        title="Executive Summary",
    )
    assert session._state["latest_generated_content"] == "# Executive Summary\n\nFresh summary from transcript."
    assert session._state["latest_generated_title"] == "Executive Summary"
    assert session._state["pending_save_to_siyuan"] is False


def test_langgraph_chat_session_save_to_siyuan_requires_session_id(tmp_path: Path) -> None:
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
            if next_node == "tool_save_to_siyuan":
                state = self._nodes[next_node](state)
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

        async def route(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            latest_session_id: str = "",
            latest_generated_content: str = "",
        ) -> str:
            return "tool_save_to_siyuan"

        async def respond_fast(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            self.last_reply_model = "openai:gemma4:e4b"
            return tool_output

        async def respond_deep(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
            raise AssertionError("Deep responder should not run in this test.")

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch("pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=FakeChatAgent()),
        patch("pawn_agent.core.langgraph_chat.build_phoenix_tracer", return_value=None),
        patch(
            "pawn_agent.core.langgraph_tools.run_save_to_siyuan_tool",
            side_effect=AssertionError("Save tool should not run without a session id."),
        ),
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        session._state["latest_generated_content"] = "# Executive Report\n\nHello"
        reply = asyncio.run(session.handle_user_input("save the analysis into siyuan"))

    assert reply == (
        "I need a session in focus before I can save to SiYuan. "
        "Retrieve or analyze a session first."
    )


def test_langgraph_chat_session_save_to_siyuan_requires_generated_content(tmp_path: Path) -> None:
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
            if next_node == "tool_save_to_siyuan":
                state = self._nodes[next_node](state)
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

        async def route(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            latest_session_id: str = "",
            latest_generated_content: str = "",
        ) -> str:
            assert latest_session_id == "oci-20260416"
            return "tool_save_to_siyuan"

        async def respond_fast(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            self.last_reply_model = "openai:gemma4:e4b"
            return tool_output

        async def respond_deep(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
            raise AssertionError("Deep responder should not run in this test.")

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch("pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=FakeChatAgent()),
        patch("pawn_agent.core.langgraph_chat.build_phoenix_tracer", return_value=None),
        patch(
            "pawn_agent.core.langgraph_tools.run_save_to_siyuan_tool",
            side_effect=AssertionError("Save tool should not run without generated content."),
        ),
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        session._state["latest_session_id"] = "oci-20260416"
        reply = asyncio.run(session.handle_user_input("save the analysis into siyuan"))

    assert reply == (
        "I need generated content to save to SiYuan. "
        "Ask me to produce the report or analysis first."
    )


def test_langgraph_chat_session_reports_missing_session_id_for_session_tool(tmp_path: Path) -> None:
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
            if next_node == "extract_session_id":
                state = await self._nodes[next_node](state)
                next_node = "tool_query_conversation"
            if next_node == "tool_query_conversation":
                state = self._nodes[next_node](state)
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

        async def route(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            latest_session_id: str = "",
            latest_generated_content: str = "",
        ) -> str:
            assert latest_session_id == ""
            self.last_route_kind = "tool_query_conversation"
            return "tool_query_conversation"

        async def extract_requested_session_id(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            latest_session_id: str = "",
        ) -> str:
            return ""

        async def respond_fast(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            self.last_reply_model = "openai:gemma4:e4b"
            return tool_output

        async def respond_deep(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
            raise AssertionError("Deep responder should not run in this test.")

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch("pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=FakeChatAgent()),
        patch("pawn_agent.core.langgraph_chat.build_phoenix_tracer", return_value=None),
        patch(
            "pawn_agent.core.langgraph_tools.run_query_conversation_tool",
            side_effect=AssertionError("Tool should not run without a session id."),
        ),
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        reply = asyncio.run(session.handle_user_input("extract action points"))

    assert reply == (
        "I need a session id to retrieve a conversation. "
        "Ask for available sessions first or specify the session id."
    )
    assert session._state["latest_session_id"] == ""
    assert session._state["requested_session_id"] == ""


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
            if next_node == "extract_session_id":
                state = await self._nodes[next_node](state)
                next_node = "tool_query_conversation"
            if next_node in {"tool_list_sessions", "tool_query_conversation"}:
                state = self._nodes[next_node](state)
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

        async def route(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            latest_session_id: str = "",
            latest_generated_content: str = "",
        ) -> str:
            assert user_prompt == "show latest sessions"
            assert chat_history == [{"role": "user", "content": "show latest sessions"}]
            assert latest_session_id == ""
            self.last_route_kind = "tool_list_sessions"
            self.last_tool_name = ""
            return "tool_list_sessions"

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
        patch(
            "pawn_agent.core.langgraph_tools.run_list_sessions_tool",
            return_value="Found 1 session(s):\n  sess-123",
        ),
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


def test_langgraph_chat_session_emits_save_to_siyuan_spans(tmp_path: Path) -> None:
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
            if next_node == "tool_save_to_siyuan":
                state = self._nodes[next_node](state)
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

        async def route(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            latest_session_id: str = "",
            latest_generated_content: str = "",
        ) -> str:
            assert latest_session_id == "oci-20260416"
            assert latest_generated_content.startswith("# Executive Report")
            self.last_route_kind = "tool_save_to_siyuan"
            return "tool_save_to_siyuan"

        async def respond_fast(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            self.last_reply_model = "openai:gemma4:e4b"
            return f"Saved to SiYuan: {tool_output}"

        async def respond_deep(self, user_prompt: str, chat_history: list[dict[str, str]]) -> str:
            raise AssertionError("Deep responder should not run in save path test.")

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch("pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=FakeChatAgent()),
        patch("pawn_agent.core.langgraph_chat.build_phoenix_tracer", return_value=FakeTracer()),
        patch(
            "pawn_agent.core.langgraph_tools.run_save_to_siyuan_tool",
            return_value="siyuan://blocks/doc-123",
        ),
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        session._state["latest_session_id"] = "oci-20260416"
        session._state["latest_generated_content"] = "# Executive Report\n\nHello"
        session._state["latest_generated_title"] = "Executive Report"
        reply = asyncio.run(session.handle_user_input("save the analysis into siyuan"))

    assert reply == "Saved to SiYuan: siyuan://blocks/doc-123"
    span_names = [event[1] for event in span_events if event[0] == "enter"]
    assert span_names == [
        "langgraph-chat-turn",
        "langgraph-human-input",
        "langgraph-route",
        "langgraph-tool-save-to-siyuan",
        "langgraph-respond-fast",
    ]
    assert ("attr", "langgraph-tool-save-to-siyuan", "tool.name", "save_to_siyuan") in span_events
    assert ("attr", "langgraph-tool-save-to-siyuan", "session.id", "oci-20260416") in span_events
    assert ("attr", "langgraph-chat-turn", "tool.name", "save_to_siyuan") in span_events
