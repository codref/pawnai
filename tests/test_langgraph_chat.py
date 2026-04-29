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
    build_tool_analyze_summary_node,
    build_tool_query_conversation_node,
    resolve_session_id,
    resolve_session_id_from_list_output,
    run_analyze_summary_tool,
    run_list_sessions_tool,
    run_query_conversation_tool,
    run_save_to_siyuan_tool,
)
from pawn_agent.tools.analyze_summary import analyze_summary_impl
from pawn_agent.tools.analyze_summary import build as build_analyze_summary_tool
from pawn_agent.tools.list_sessions import build as build_list_sessions_tool
from pawn_agent.tools.list_sessions import list_sessions_impl
from pawn_agent.tools.query_conversation import build as build_query_conversation_tool
from pawn_agent.tools.query_conversation import query_conversation_impl
from pawn_agent.tools.save_to_siyuan import build as build_save_to_siyuan_tool
from pawn_agent.tools.save_to_siyuan import save_to_siyuan_impl
from pawn_agent.utils.config import load_config


def test_new_langgraph_chat_state_resets_history() -> None:
    assert new_langgraph_chat_state() == {
        "session_state": {
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
            "recent_memories": [],
        },
        "durable_facts": {
            "latest_session_id": "",
        },
        "artifacts": {
            "tool_output": "",
            "latest_generated_content": "",
            "latest_generated_title": "",
            "session_catalog_output": "",
            "latest_session_transcript": "",
        },
        "recent_messages": [],
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
        cfg=SimpleNamespace(),
        last_route_kind="reply_fast",
        last_route_model="openai:gemma4:e4b",
        last_reply_model="openai:gemma4:e4b",
        last_tool_name="",
        last_action_plan=["reply_fast"],
    )
    with patch(
        "pawn_agent.core.langgraph_chat._import_langgraph_core",
        return_value=(FakeStateGraph, "__start__", "__end__"),
    ):
        compiled = asyncio.run(build_langgraph_chat_graph(chat_agent=fake_agent))

    assert isinstance(compiled, FakeCompiledGraph)
    assert set(calls["nodes"]) == {
        "human_input",
        "recall_memories",
        "plan",
        "dispatch",
        "extract_session_id",
        "tool_list_sessions",
        "tool_analyze_summary",
        "tool_query_conversation",
        "tool_save_to_siyuan",
        "tool_memorize",
        "tool_recall_memory",
        "tool_search_knowledge",
        "tool_vectorize",
        "tool_push_queue_message",
        "respond_fast",
        "respond_deep",
    }
    assert calls["edges"] == [
        ("__start__", "human_input"),
        ("human_input", "recall_memories"),
        ("recall_memories", "plan"),
        ("plan", "dispatch"),
        ("tool_list_sessions", "dispatch"),
        ("tool_analyze_summary", "dispatch"),
        ("tool_query_conversation", "dispatch"),
        ("tool_save_to_siyuan", "dispatch"),
        ("tool_memorize", "dispatch"),
        ("tool_recall_memory", "dispatch"),
        ("tool_search_knowledge", "dispatch"),
        ("tool_vectorize", "dispatch"),
        ("tool_push_queue_message", "dispatch"),
        ("respond_fast", "dispatch"),
        ("respond_deep", "dispatch"),
    ]
    assert calls["conditionals"][0][0] == "extract_session_id"
    assert calls["conditionals"][0][2] == {
        "tool_analyze_summary": "tool_analyze_summary",
        "tool_query_conversation": "tool_query_conversation",
    }
    assert calls["conditionals"][1][0] == "dispatch"
    assert calls["conditionals"][1][2] == {
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
        "__end__": "__end__",
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
        patch(
            "pawn_agent.core.langgraph_chat._import_phoenix_register", return_value=fake_register
        ),
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


def test_serialize_langgraph_state_returns_json_snapshot() -> None:
    from pawn_agent.core.langgraph_state import serialize_langgraph_state

    state_json = serialize_langgraph_state(new_langgraph_chat_state())

    assert '"session_state"' in state_json
    assert '"durable_facts"' in state_json
    assert '"artifacts"' in state_json
    assert '"recent_messages"' in state_json


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


def test_analyze_summary_impl_delegates_to_shared_helper() -> None:
    cfg = SimpleNamespace(db_dsn="postgresql://dummy")

    with (
        patch(
            "pawn_agent.tools.analyze_summary.run_analysis",
            return_value="## Title\nDemo",
        ) as mock_run,
        patch(
            "pawn_agent.tools.analyze_summary.get_session_analysis",
            return_value=SimpleNamespace(
                title="Demo",
                tags=["demo"],
                sentiment_tags=["collaborative"],
            ),
        ),
        patch(
            "pawn_agent.tools.analyze_summary.do_save_to_siyuan",
            return_value="siyuan://blocks/doc-123",
        ) as mock_save,
    ):
        reply = asyncio.run(analyze_summary_impl(cfg, "sess-123", save=True, title=None))

    assert reply.startswith("Analysis saved to database and SiYuan")
    mock_run.assert_called_once_with(cfg, "sess-123")
    mock_save.assert_called_once()


def test_analyze_summary_build_delegates_to_shared_helper() -> None:
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
            "pawn_agent.tools.analyze_summary.analyze_summary_impl",
            return_value="## Title\nDemo",
        ) as mock_impl,
    ):
        tool_obj = build_analyze_summary_tool(cfg)
        reply = asyncio.run(captured["fn"]("sess-123", save=False, title=None))

    assert isinstance(tool_obj, FakeTool)
    assert reply == "## Title\nDemo"
    mock_impl.assert_called_once_with(cfg, "sess-123", save=False, title=None)


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
            if user_prompt.startswith("You are a planning model"):
                return '["reply_fast"]'
            return "fast:hello"

    with patch("pawn_agent.core.langgraph_chat.PlainPydanticChatAgent", FakeAgent):
        agent = LangGraphRouterChatAgent(cfg)
        route = asyncio.run(agent.plan("hello", [{"role": "user", "content": "hello"}]))
        reply = asyncio.run(agent.respond_fast("hello", [{"role": "user", "content": "hello"}]))

    assert route == ["reply_fast"]
    assert reply == "fast:hello"
    assert calls[0][0] == "openai:gemma4:e4b"
    assert calls[1][0] == "openai:gemma4:e4b"
    assert calls[1][1] == "hello"
    assert agent.last_action_plan == ["reply_fast"]
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
            if user_prompt.startswith("You are a planning model"):
                return '["reply_deep"]'
            return "deep:analysis"

    with patch("pawn_agent.core.langgraph_chat.PlainPydanticChatAgent", FakeAgent):
        agent = LangGraphRouterChatAgent(cfg)
        route = asyncio.run(agent.plan("analyze this deeply", []))
        reply = asyncio.run(agent.respond_deep("analyze this deeply", []))

    assert route == ["reply_deep"]
    assert reply == "deep:analysis"
    assert calls[0][0] == "openai:gemma4:e4b"
    assert calls[1][0] == "openai:gemma4:26b"
    assert agent.last_action_plan == ["reply_deep"]
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
            if user_prompt.startswith("You are a planning model"):
                return '["tool_list_sessions", "reply_fast"]'
            return "Here are the latest sessions."

    with (
        patch("pawn_agent.core.langgraph_chat.PlainPydanticChatAgent", FakeAgent),
        patch(
            "pawn_agent.core.langgraph_tools.list_sessions_impl",
            return_value="Found 1 session(s):\n  sess-123",
        ) as mock_tool,
    ):
        agent = LangGraphRouterChatAgent(cfg)
        route = asyncio.run(agent.plan("show me the latest sessions", []))
        tool_output = run_list_sessions_tool(cfg)
        reply = asyncio.run(
            agent.respond_fast(
                "show me the latest sessions",
                [],
                tool_output=tool_output,
            )
        )

    assert route == ["tool_list_sessions", "reply_fast"]
    assert reply == "Here are the latest sessions."
    assert calls[0][0] == "openai:gemma4:e4b"
    assert calls[1][0] == "openai:gemma4:e4b"
    assert "Tool result:\nFound 1 session(s):\n  sess-123" in calls[1][1]
    mock_tool.assert_called_once_with(cfg)
    assert agent.last_action_plan == ["tool_list_sessions", "reply_fast"]
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
            if user_prompt.startswith("You are a planning model"):
                return '["tool_query_conversation", "reply_fast"]'
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
            agent.plan(
                "show session sess-123",
                [],
            )
        )
        session_id = asyncio.run(agent.extract_requested_session_id("show session sess-123", []))
        tool_output = run_query_conversation_tool(cfg, session_id or "")
        reply = asyncio.run(
            agent.respond_fast(
                "show session sess-123",
                [],
                tool_output=tool_output,
            )
        )

    assert route == ["tool_query_conversation", "reply_fast"]
    assert session_id == "sess-123"
    assert reply == "Here is the transcript you asked for."
    mock_tool.assert_called_once_with(cfg, "sess-123")
    assert "Tool result:\nSpeaker A: hi\nSpeaker B: hello" in calls[2][1]


def test_langgraph_router_agent_routes_to_analyze_summary_tool(tmp_path: Path) -> None:
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
            if user_prompt.startswith("You are a planning model"):
                return '["tool_analyze_summary", "reply_fast"]'
            return "## Title\nStandard Analysis"

    with (
        patch("pawn_agent.core.langgraph_chat.PlainPydanticChatAgent", FakeAgent),
        patch(
            "pawn_agent.core.langgraph_tools.analyze_summary_impl",
            return_value="## Title\nStandard Analysis",
        ) as mock_tool,
    ):
        agent = LangGraphRouterChatAgent(cfg)
        route = asyncio.run(agent.plan("run the standard analysis for session sess-123", []))
        tool_output = asyncio.run(run_analyze_summary_tool(cfg, "sess-123"))
        reply = asyncio.run(
            agent.respond_fast(
                "run the standard analysis for session sess-123",
                [],
                tool_output=tool_output,
            )
        )

    assert route == ["tool_analyze_summary", "reply_fast"]
    assert reply == "## Title\nStandard Analysis"
    mock_tool.assert_called_once_with(cfg, "sess-123", save=False, title=None)
    assert "tool_analyze_summary" in calls[0][1]


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
            if user_prompt.startswith("You are a planning model"):
                return '["tool_save_to_siyuan", "reply_fast"]'
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
            agent.plan(
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

    assert route == ["tool_save_to_siyuan", "reply_fast"]
    assert reply == "Saved the report to SiYuan."
    mock_tool.assert_called_once_with(
        cfg,
        session_id="sess-123",
        content="# Report\n\nHello",
        title="Report",
        path=None,
    )
    assert "Session transcript cached:\nno" in calls[0][1]
    assert (
        "Latest generated content available (for context only — do NOT auto-save):\nyes"
        in calls[0][1]
    )
    assert "tool_save_to_siyuan" in calls[0][1]


def test_langgraph_session_tool_helper_prefers_current_turn_id() -> None:
    cfg = SimpleNamespace()
    state = {
        "session_state": {"requested_session_id": "sess-456"},
        "durable_facts": {"latest_session_id": "sess-123"},
        "artifacts": {},
        "recent_messages": [],
    }

    _, session_id = resolve_session_id(state, cfg, bootstrap_catalog=False)
    assert session_id == "sess-456"


def test_langgraph_session_tool_helper_falls_back_to_latest_session_id() -> None:
    cfg = SimpleNamespace()
    state = {
        "session_state": {"requested_session_id": "", "latest_user_message": "yes"},
        "durable_facts": {"latest_session_id": "sess-123"},
        "artifacts": {},
        "recent_messages": [],
    }

    _, session_id = resolve_session_id(state, cfg, bootstrap_catalog=False)
    assert session_id == "sess-123"


def test_langgraph_session_tool_helper_resolves_named_session_from_catalog() -> None:
    cfg = SimpleNamespace()
    state = {
        "session_state": {
            "requested_session_id": "tom",
            "latest_user_message": "retrieve latest tom session",
        },
        "durable_facts": {"latest_session_id": ""},
        "artifacts": {
            "session_catalog_output": (
                "Found 3 session(s):\n"
                "  oci-20260416  |  segments: 1  |  duration: 1m 00s  |  updated: 2026-04-16 15:57\n"
                "  tom-20260416  |  segments: 2  |  duration: 2m 00s  |  updated: 2026-04-16 15:41\n"
                "  tom-20260415  |  segments: 3  |  duration: 3m 00s  |  updated: 2026-04-15 16:32"
            )
        },
        "recent_messages": [],
    }

    _, session_id = resolve_session_id(state, cfg, bootstrap_catalog=False)
    assert session_id == "tom-20260416"


def test_tool_query_conversation_bootstraps_session_catalog_for_first_turn_lookup() -> None:
    cfg = SimpleNamespace()
    node = build_tool_query_conversation_node(cfg=cfg)
    state = {
        "session_state": {
            "latest_user_message": (
                "retrieve latest tom conversation available and save to siyuan "
                "a deep report ready for executive presentation"
            ),
            "requested_session_id": "",
            "action_plan": ["tool_save_to_siyuan", "reply_fast"],
        },
        "durable_facts": {"latest_session_id": ""},
        "artifacts": {
            "session_catalog_output": "",
            "tool_output": "",
        },
        "recent_messages": [],
    }
    catalog_output = (
        "Found 3 session(s):\n"
        "  oci-20260416  |  segments: 138  |  duration: 19m 53s  |  updated: 2026-04-16 15:57\n"
        "  tom-20260416  |  segments: 564  |  duration: 1h 03m 38s  |  updated: 2026-04-16 15:41\n"
        "  tom-20260415  |  segments: 706  |  duration: 1h 10m 05s  |  updated: 2026-04-15 16:32"
    )

    with (
        patch(
            "pawn_agent.core.langgraph_tools.run_list_sessions_tool",
            return_value=catalog_output,
        ) as mock_list,
        patch(
            "pawn_agent.core.langgraph_tools.run_query_conversation_tool",
            return_value="Speaker A: hi\nSpeaker B: hello",
        ) as mock_query,
    ):
        updated = node(state)

    assert updated["durable_facts"]["latest_session_id"] == "tom-20260416"
    assert updated["artifacts"]["session_catalog_output"] == catalog_output
    assert updated["artifacts"]["tool_output"] == "Speaker A: hi\nSpeaker B: hello"
    assert updated["session_state"]["tool_name"] == "query_conversation"
    assert updated["session_state"]["action_plan"] == ["tool_save_to_siyuan", "reply_fast"]
    mock_list.assert_called_once_with(cfg)
    mock_query.assert_called_once_with(cfg, "tom-20260416")


def test_tool_analyze_summary_bootstraps_session_catalog_for_first_turn_lookup() -> None:
    cfg = SimpleNamespace()
    node = build_tool_analyze_summary_node(cfg=cfg)
    state = {
        "session_state": {
            "latest_user_message": "run the standard analysis for the latest tom session",
            "requested_session_id": "",
        },
        "durable_facts": {"latest_session_id": ""},
        "artifacts": {
            "session_catalog_output": "",
            "tool_output": "",
        },
        "recent_messages": [],
    }
    catalog_output = (
        "Found 3 session(s):\n"
        "  oci-20260416  |  segments: 138  |  duration: 19m 53s  |  updated: 2026-04-16 15:57\n"
        "  tom-20260416  |  segments: 564  |  duration: 1h 03m 38s  |  updated: 2026-04-16 15:41\n"
        "  tom-20260415  |  segments: 706  |  duration: 1h 10m 05s  |  updated: 2026-04-15 16:32"
    )

    with (
        patch(
            "pawn_agent.core.langgraph_tools.run_list_sessions_tool",
            return_value=catalog_output,
        ) as mock_list,
        patch(
            "pawn_agent.core.langgraph_tools.run_analyze_summary_tool",
            return_value="## Title\nStandard Analysis",
        ) as mock_analyze,
    ):
        updated = asyncio.run(node(state))

    assert updated["durable_facts"]["latest_session_id"] == "tom-20260416"
    assert updated["artifacts"]["session_catalog_output"] == catalog_output
    assert updated["artifacts"]["tool_output"] == "## Title\nStandard Analysis"
    assert updated["session_state"]["tool_name"] == "analyze_summary"
    mock_list.assert_called_once_with(cfg)
    mock_analyze.assert_called_once_with(cfg, "tom-20260416")


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


def test_langgraph_router_agent_session_id_extractor_returns_none_when_absent(
    tmp_path: Path,
) -> None:
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
        session_id = asyncio.run(agent.extract_requested_session_id("extract action points", []))

    assert session_id == ""


def test_langgraph_chat_session_uses_deep_responder_after_query_for_executive_report(
    tmp_path: Path,
) -> None:
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
            import asyncio as _asyncio

            state = dict(payload)
            state = self._nodes["human_input"](state)
            state = await self._nodes["plan"](state)
            while True:
                state = self._nodes["dispatch"](state)
                next_node = self._route_fn(state)
                if next_node == "__end__":
                    break
                if next_node == "extract_session_id":
                    state = await self._nodes["extract_session_id"](state)
                    route_kind = (state.get("session_state") or {}).get("route_kind", "")
                    tool_name = (
                        "tool_analyze_summary"
                        if route_kind == "tool_analyze_summary"
                        else "tool_query_conversation"
                    )
                    state = self._nodes[tool_name](state)
                else:
                    node_fn = self._nodes[next_node]
                    state = (
                        await node_fn(state)
                        if _asyncio.iscoroutinefunction(node_fn)
                        else node_fn(state)
                    )
            return state

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
        cfg = None
        last_route_kind = ""
        last_route_model = "openai:gemma4:e4b"
        last_reply_model = ""
        last_tool_name = ""

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
            assert "deep report ready for executive presentation" in user_prompt
            self.last_route_kind = "tool_query_conversation"
            return ["tool_query_conversation", "reply_deep"]

        async def extract_requested_session_id(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            latest_session_id: str = "",
        ) -> str:
            return "tom-20260416"

        async def respond_fast(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            raise AssertionError("Fast responder should not run for this deep report flow.")

        async def respond_deep(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            self.last_reply_model = "openai:gemma4:26b"
            assert user_prompt == (
                "retrieve latest tom conversation available and give me a deep report "
                "ready for executive presentation"
            )
            assert tool_output == "Transcript for tom-20260416"
            return "# Executive Report\n\nDeep analysis from transcript."

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
            "pawn_agent.core.langgraph_tools.run_query_conversation_tool",
            return_value="Transcript for tom-20260416",
        ),
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        reply = asyncio.run(
            session.handle_user_input(
                "retrieve latest tom conversation available and give me a deep report "
                "ready for executive presentation"
            )
        )

    assert reply == "# Executive Report\n\nDeep analysis from transcript."
    assert session._state["session_state"]["reply_model"] == "openai:gemma4:26b"
    assert session._state["durable_facts"]["latest_session_id"] == "tom-20260416"


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


def test_langgraph_router_prompt_guides_followup_report_to_conversation_tool(
    tmp_path: Path,
) -> None:
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
        prompt = agent._planner_prompt(
            "give me an executive report",
            [{"role": "assistant", "content": "I retrieved session oci-20260416."}],
            latest_session_id="oci-20260416",
        )

    assert "executive report" in prompt
    assert "tool_query_conversation" in prompt
    assert "reply_deep" in prompt
    assert "summary" in prompt
    assert "report" in prompt
    assert "tool_save_to_siyuan" in prompt


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
            history = list(payload.get("recent_messages", []))
            incoming_prompt = payload["session_state"]["incoming_prompt"]
            history.append({"role": "user", "content": incoming_prompt})
            history.append({"role": "assistant", "content": f"echo:{incoming_prompt}"})
            return {
                "session_state": {
                    "incoming_prompt": "",
                    "latest_user_message": incoming_prompt,
                    "latest_assistant_message": f"echo:{incoming_prompt}",
                    "turn_count": int(payload["session_state"].get("turn_count", 0)) + 1,
                },
                "recent_messages": history,
                "durable_facts": {},
                "artifacts": {},
            }

    with (
        patch("pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=object()),
        patch(
            "pawn_agent.core.langgraph_chat.build_langgraph_chat_graph", return_value=FakeGraph()
        ),
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        first = asyncio.run(session.handle_user_input("hello"))
        second = asyncio.run(session.handle_user_input("again"))
        asyncio.run(session.reset())

    assert first == "echo:hello"
    assert second == "echo:again"
    assert payloads[0]["recent_messages"] == []
    assert payloads[1]["recent_messages"] == [
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
            import asyncio as _asyncio

            state = dict(payload)
            state = self._nodes["human_input"](state)
            state = await self._nodes["plan"](state)
            while True:
                state = self._nodes["dispatch"](state)
                next_node = self._route_fn(state)
                if next_node == "__end__":
                    break
                if next_node == "extract_session_id":
                    state = await self._nodes["extract_session_id"](state)
                    route_kind = (state.get("session_state") or {}).get("route_kind", "")
                    tool_name = (
                        "tool_analyze_summary"
                        if route_kind == "tool_analyze_summary"
                        else "tool_query_conversation"
                    )
                    state = self._nodes[tool_name](state)
                else:
                    node_fn = self._nodes[next_node]
                    state = (
                        await node_fn(state)
                        if _asyncio.iscoroutinefunction(node_fn)
                        else node_fn(state)
                    )
            return state

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
        cfg = None
        last_route_kind = ""
        last_route_model = "openai:gemma4:e4b"
        last_reply_model = ""
        last_tool_name = ""
        tool_session_ids: list[str] = []

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
            if user_prompt == "extract action points":
                assert latest_session_id == "oci-20260416"
            self.last_route_kind = "tool_query_conversation"
            return ["tool_query_conversation", "reply_fast"]

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

        async def respond_deep(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            raise AssertionError("Deep responder should not run in this test.")

    fake_chat_agent = FakeChatAgent()

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch(
            "pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=fake_chat_agent
        ),
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
    assert session._state["durable_facts"]["latest_session_id"] == "oci-20260417"
    assert session._state["session_state"]["requested_session_id"] == ""
    assert session._state["artifacts"]["tool_output"] == ""


def test_langgraph_chat_session_promotes_latest_session_id_from_list_results(
    tmp_path: Path,
) -> None:
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
            import asyncio as _asyncio

            state = dict(payload)
            state = self._nodes["human_input"](state)
            state = await self._nodes["plan"](state)
            while True:
                state = self._nodes["dispatch"](state)
                next_node = self._route_fn(state)
                if next_node == "__end__":
                    break
                if next_node == "extract_session_id":
                    state = await self._nodes["extract_session_id"](state)
                    route_kind = (state.get("session_state") or {}).get("route_kind", "")
                    tool_name = (
                        "tool_analyze_summary"
                        if route_kind == "tool_analyze_summary"
                        else "tool_query_conversation"
                    )
                    state = self._nodes[tool_name](state)
                else:
                    node_fn = self._nodes[next_node]
                    state = (
                        await node_fn(state)
                        if _asyncio.iscoroutinefunction(node_fn)
                        else node_fn(state)
                    )
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
            if start == "dispatch":
                self._route_fn = fn

        def compile(self):
            graph = FakeCompiledGraph(self._nodes)
            graph.set_route_fn(self._route_fn)
            return graph

    class FakeChatAgent:
        cfg = None
        last_route_kind = ""
        last_route_model = "openai:gemma4:e4b"
        last_reply_model = ""
        last_tool_name = ""

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
            if user_prompt == "retrieve last oci session":
                assert latest_session_id == ""
                return ["tool_list_sessions", "reply_fast"]
            if user_prompt == "extract an executive summary":
                assert latest_session_id == "oci-20260416"
                return ["tool_query_conversation", "reply_fast"]
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

        async def respond_deep(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
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
    assert session._state["durable_facts"]["latest_session_id"] == "oci-20260416"


def test_langgraph_chat_session_resolves_named_session_and_confirmation_from_catalog(
    tmp_path: Path,
) -> None:
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
            import asyncio as _asyncio

            state = dict(payload)
            state = self._nodes["human_input"](state)
            state = await self._nodes["plan"](state)
            while True:
                state = self._nodes["dispatch"](state)
                next_node = self._route_fn(state)
                if next_node == "__end__":
                    break
                if next_node == "extract_session_id":
                    state = await self._nodes["extract_session_id"](state)
                    route_kind = (state.get("session_state") or {}).get("route_kind", "")
                    tool_name = (
                        "tool_analyze_summary"
                        if route_kind == "tool_analyze_summary"
                        else "tool_query_conversation"
                    )
                    state = self._nodes[tool_name](state)
                else:
                    node_fn = self._nodes[next_node]
                    state = (
                        await node_fn(state)
                        if _asyncio.iscoroutinefunction(node_fn)
                        else node_fn(state)
                    )
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
            if start == "dispatch":
                self._route_fn = fn

        def compile(self):
            graph = FakeCompiledGraph(self._nodes)
            graph.set_route_fn(self._route_fn)
            return graph

    class FakeChatAgent:
        cfg = None
        last_route_kind = ""
        last_route_model = "openai:gemma4:e4b"
        last_reply_model = ""
        last_tool_name = ""

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
            if user_prompt == "list latest sessions":
                return ["tool_list_sessions", "reply_fast"]
            if user_prompt in {"retrieve latest tom session", "yes"}:
                return ["tool_query_conversation", "reply_fast"]
            raise AssertionError(f"Unexpected prompt: {user_prompt}")

        async def extract_requested_session_id(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            latest_session_id: str = "",
        ) -> str:
            if user_prompt == "retrieve latest tom session":
                return "tom"
            if user_prompt == "yes":
                assert latest_session_id == "tom-20260416"
                return ""
            raise AssertionError(f"Unexpected prompt: {user_prompt}")

        async def respond_fast(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            self.last_reply_model = "openai:gemma4:e4b"
            if user_prompt == "list latest sessions":
                return "Here are the latest sessions."
            if user_prompt == "retrieve latest tom session":
                assert tool_output == "Transcript for tom-20260416"
                return "Loaded tom-20260416."
            if user_prompt == "yes":
                assert tool_output == "Transcript for tom-20260416"
                return "Loaded tom-20260416 again."
            raise AssertionError(f"Unexpected prompt: {user_prompt}")

        async def respond_deep(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            raise AssertionError("Deep responder should not run in this test.")

    queried_ids: list[str] = []

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
                "Found 3 session(s):\n"
                "  oci-20260416  |  segments: 1  |  duration: 1m 00s  |  updated: 2026-04-16 15:57\n"
                "  tom-20260416  |  segments: 2  |  duration: 2m 00s  |  updated: 2026-04-16 15:41\n"
                "  tom-20260415  |  segments: 3  |  duration: 3m 00s  |  updated: 2026-04-15 16:32"
            ),
        ),
        patch(
            "pawn_agent.core.langgraph_tools.run_query_conversation_tool",
            side_effect=lambda cfg, session_id: queried_ids.append(session_id)
            or f"Transcript for {session_id}",
        ),
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        first = asyncio.run(session.handle_user_input("list latest sessions"))
        second = asyncio.run(session.handle_user_input("retrieve latest tom session"))
        third = asyncio.run(session.handle_user_input("yes"))

    assert first == "Here are the latest sessions."
    assert second == "Loaded tom-20260416."
    assert third == "Loaded tom-20260416 again."
    assert queried_ids == ["tom-20260416", "tom-20260416"]
    assert session._state["durable_facts"]["latest_session_id"] == "tom-20260416"


def test_langgraph_chat_session_preserves_session_focus_after_transcript_summary(
    tmp_path: Path,
) -> None:
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
            import asyncio as _asyncio

            state = dict(payload)
            state = self._nodes["human_input"](state)
            state = await self._nodes["plan"](state)
            while True:
                state = self._nodes["dispatch"](state)
                next_node = self._route_fn(state)
                if next_node == "__end__":
                    break
                if next_node == "extract_session_id":
                    state = await self._nodes["extract_session_id"](state)
                    route_kind = (state.get("session_state") or {}).get("route_kind", "")
                    tool_name = (
                        "tool_analyze_summary"
                        if route_kind == "tool_analyze_summary"
                        else "tool_query_conversation"
                    )
                    state = self._nodes[tool_name](state)
                else:
                    node_fn = self._nodes[next_node]
                    state = (
                        await node_fn(state)
                        if _asyncio.iscoroutinefunction(node_fn)
                        else node_fn(state)
                    )
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
            if start == "dispatch":
                self._route_fn = fn

        def compile(self):
            graph = FakeCompiledGraph(self._nodes)
            graph.set_route_fn(self._route_fn)
            return graph

    class FakeChatAgent:
        cfg = None
        last_route_kind = ""
        last_route_model = "openai:gemma4:e4b"
        last_reply_model = ""
        last_tool_name = ""

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
            if user_prompt == "retrieve latest tom session":
                return ["tool_query_conversation", "reply_fast"]
            if user_prompt == "save a deep analysis of the conversation in siyuan":
                assert latest_session_id == "tom-20260416"
                assert latest_generated_content == "Summary of tom-20260416."
                return ["tool_save_to_siyuan", "reply_fast"]
            raise AssertionError(f"Unexpected prompt: {user_prompt}")

        async def extract_requested_session_id(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            latest_session_id: str = "",
        ) -> str:
            return "tom-20260416"

        async def respond_fast(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            self.last_reply_model = "openai:gemma4:e4b"
            if user_prompt == "retrieve latest tom session":
                assert tool_output == "Transcript for tom-20260416"
                return "Summary of tom-20260416."
            if user_prompt == "save a deep analysis of the conversation in siyuan":
                return f"Saved to SiYuan: {tool_output}"
            raise AssertionError(f"Unexpected prompt: {user_prompt}")

        async def respond_deep(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
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
            "pawn_agent.core.langgraph_tools.run_query_conversation_tool",
            return_value="Transcript for tom-20260416",
        ),
        patch(
            "pawn_agent.core.langgraph_tools.run_save_to_siyuan_tool",
            return_value="siyuan://blocks/doc-123",
        ),
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        first = asyncio.run(session.handle_user_input("retrieve latest tom session"))
        second = asyncio.run(
            session.handle_user_input("save a deep analysis of the conversation in siyuan")
        )

    assert first == "Summary of tom-20260416."
    assert second == "Saved to SiYuan: siyuan://blocks/doc-123"
    assert session._state["durable_facts"]["latest_session_id"] == "tom-20260416"


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
            import asyncio as _asyncio

            state = dict(payload)
            state = self._nodes["human_input"](state)
            state = await self._nodes["plan"](state)
            while True:
                state = self._nodes["dispatch"](state)
                next_node = self._route_fn(state)
                if next_node == "__end__":
                    break
                if next_node == "extract_session_id":
                    state = await self._nodes["extract_session_id"](state)
                    route_kind = (state.get("session_state") or {}).get("route_kind", "")
                    tool_name = (
                        "tool_analyze_summary"
                        if route_kind == "tool_analyze_summary"
                        else "tool_query_conversation"
                    )
                    state = self._nodes[tool_name](state)
                else:
                    node_fn = self._nodes[next_node]
                    state = (
                        await node_fn(state)
                        if _asyncio.iscoroutinefunction(node_fn)
                        else node_fn(state)
                    )
            return state

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
        cfg = None
        last_route_kind = ""
        last_route_model = "openai:gemma4:e4b"
        last_reply_model = ""
        last_tool_name = ""

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
            if user_prompt == "show session oci-20260416":
                return ["tool_query_conversation", "reply_fast"]
            if user_prompt == "give me an executive report":
                assert latest_session_id == "oci-20260416"
                return ["tool_query_conversation", "reply_fast"]
            if user_prompt == "save the analysis into siyuan":
                assert latest_session_id == "oci-20260416"
                assert latest_generated_content.startswith("# Executive Report")
                return ["tool_save_to_siyuan", "reply_fast"]
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

        async def respond_deep(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            raise AssertionError("Deep responder should not run in this test.")

    FakeChatAgent.cfg = cfg
    fake_chat_agent = FakeChatAgent()

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch(
            "pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=fake_chat_agent
        ),
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
    assert (
        session._state["artifacts"]["latest_generated_content"]
        == "# Executive Report\n\nThis is the report."
    )
    assert session._state["artifacts"]["latest_generated_title"] == "Executive Report"
    assert session._state["durable_facts"]["latest_session_id"] == "oci-20260416"


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
            import asyncio as _asyncio

            state = dict(payload)
            state = self._nodes["human_input"](state)
            state = await self._nodes["plan"](state)
            while True:
                state = self._nodes["dispatch"](state)
                next_node = self._route_fn(state)
                if next_node == "__end__":
                    break
                if next_node == "extract_session_id":
                    state = await self._nodes["extract_session_id"](state)
                    route_kind = (state.get("session_state") or {}).get("route_kind", "")
                    tool_name = (
                        "tool_analyze_summary"
                        if route_kind == "tool_analyze_summary"
                        else "tool_query_conversation"
                    )
                    state = self._nodes[tool_name](state)
                else:
                    node_fn = self._nodes[next_node]
                    state = (
                        await node_fn(state)
                        if _asyncio.iscoroutinefunction(node_fn)
                        else node_fn(state)
                    )
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
            if start == "dispatch":
                self._route_fn = fn

        def compile(self):
            graph = FakeCompiledGraph(self._nodes)
            graph.set_route_fn(self._route_fn)
            return graph

    class FakeChatAgent:
        cfg = None
        last_route_kind = ""
        last_route_model = "openai:gemma4:e4b"
        last_reply_model = ""
        last_tool_name = ""

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
            assert user_prompt == "extract an executive summary and save it to siyuan"
            assert latest_session_id == "oci-20260416"
            assert latest_generated_content == "# Old Summary\n\nStale"
            self.last_route_kind = "tool_query_conversation"
            return ["tool_query_conversation", "reply_fast", "tool_save_to_siyuan", "reply_fast"]

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

        async def respond_deep(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            raise AssertionError("Deep responder should not run in this test.")

    FakeChatAgent.cfg = cfg
    fake_chat_agent = FakeChatAgent()

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch(
            "pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=fake_chat_agent
        ),
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
        session._state["durable_facts"]["latest_session_id"] = "oci-20260416"
        session._state["artifacts"]["latest_generated_content"] = "# Old Summary\n\nStale"
        session._state["artifacts"]["latest_generated_title"] = "Old Summary"
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
    assert (
        session._state["artifacts"]["latest_generated_content"]
        == "# Executive Summary\n\nFresh summary from transcript."
    )
    assert session._state["artifacts"]["latest_generated_title"] == "Executive Summary"


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
            import asyncio as _asyncio

            state = dict(payload)
            state = self._nodes["human_input"](state)
            state = await self._nodes["plan"](state)
            while True:
                state = self._nodes["dispatch"](state)
                next_node = self._route_fn(state)
                if next_node == "__end__":
                    break
                if next_node == "extract_session_id":
                    state = await self._nodes["extract_session_id"](state)
                    route_kind = (state.get("session_state") or {}).get("route_kind", "")
                    tool_name = (
                        "tool_analyze_summary"
                        if route_kind == "tool_analyze_summary"
                        else "tool_query_conversation"
                    )
                    state = self._nodes[tool_name](state)
                else:
                    node_fn = self._nodes[next_node]
                    state = (
                        await node_fn(state)
                        if _asyncio.iscoroutinefunction(node_fn)
                        else node_fn(state)
                    )
            return state

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
        cfg = None
        last_route_kind = ""
        last_route_model = "openai:gemma4:e4b"
        last_reply_model = ""
        last_tool_name = ""

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
            return ["tool_save_to_siyuan", "reply_fast"]

        async def respond_fast(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            self.last_reply_model = "openai:gemma4:e4b"
            return tool_output

        async def respond_deep(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            raise AssertionError("Deep responder should not run in this test.")

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch(
            "pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=FakeChatAgent()
        ),
        patch("pawn_agent.core.langgraph_chat.build_phoenix_tracer", return_value=None),
        patch(
            "pawn_agent.core.langgraph_tools.run_save_to_siyuan_tool",
            side_effect=AssertionError("Save tool should not run without a session id."),
        ),
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        session._state["artifacts"]["latest_generated_content"] = "# Executive Report\n\nHello"
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
            import asyncio as _asyncio

            state = dict(payload)
            state = self._nodes["human_input"](state)
            state = await self._nodes["plan"](state)
            while True:
                state = self._nodes["dispatch"](state)
                next_node = self._route_fn(state)
                if next_node == "__end__":
                    break
                node_fn = self._nodes[next_node]
                state = (
                    await node_fn(state)
                    if _asyncio.iscoroutinefunction(node_fn)
                    else node_fn(state)
                )
            return state

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
        cfg = None
        last_route_kind = ""
        last_route_model = "openai:gemma4:e4b"
        last_reply_model = ""
        last_tool_name = ""

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
            assert latest_session_id == "oci-20260416"
            return ["tool_save_to_siyuan", "reply_fast"]

        async def respond_fast(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            self.last_reply_model = "openai:gemma4:e4b"
            return tool_output

        async def respond_deep(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            raise AssertionError("Deep responder should not run in this test.")

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch(
            "pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=FakeChatAgent()
        ),
        patch("pawn_agent.core.langgraph_chat.build_phoenix_tracer", return_value=None),
        patch(
            "pawn_agent.core.langgraph_tools.run_save_to_siyuan_tool",
            side_effect=AssertionError("Save tool should not run without generated content."),
        ),
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        session._state["durable_facts"]["latest_session_id"] = "oci-20260416"
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
            import asyncio as _asyncio

            state = dict(payload)
            state = self._nodes["human_input"](state)
            state = await self._nodes["plan"](state)
            while True:
                state = self._nodes["dispatch"](state)
                next_node = self._route_fn(state)
                if next_node == "__end__":
                    break
                if next_node == "extract_session_id":
                    state = await self._nodes["extract_session_id"](state)
                    route_kind = (state.get("session_state") or {}).get("route_kind", "")
                    tool_name = (
                        "tool_analyze_summary"
                        if route_kind == "tool_analyze_summary"
                        else "tool_query_conversation"
                    )
                    state = self._nodes[tool_name](state)
                else:
                    node_fn = self._nodes[next_node]
                    state = (
                        await node_fn(state)
                        if _asyncio.iscoroutinefunction(node_fn)
                        else node_fn(state)
                    )
            return state

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
        cfg = None
        last_route_kind = ""
        last_route_model = "openai:gemma4:e4b"
        last_reply_model = ""
        last_tool_name = ""

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
            assert latest_session_id == ""
            self.last_route_kind = "tool_query_conversation"
            return ["tool_query_conversation", "reply_fast"]

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

        async def respond_deep(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            raise AssertionError("Deep responder should not run in this test.")

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch(
            "pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=FakeChatAgent()
        ),
        patch("pawn_agent.core.langgraph_chat.build_phoenix_tracer", return_value=None),
        patch(
            "pawn_agent.core.langgraph_tools.run_list_sessions_tool",
            return_value="",
        ),
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
    assert session._state["durable_facts"]["latest_session_id"] == ""
    assert session._state["session_state"]["requested_session_id"] == ""


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
            import asyncio as _asyncio

            state = dict(payload)
            state = self._nodes["human_input"](state)
            state = await self._nodes["plan"](state)
            while True:
                state = self._nodes["dispatch"](state)
                next_node = self._route_fn(state)
                if next_node == "__end__":
                    break
                if next_node == "extract_session_id":
                    state = await self._nodes["extract_session_id"](state)
                    route_kind = (state.get("session_state") or {}).get("route_kind", "")
                    tool_name = (
                        "tool_analyze_summary"
                        if route_kind == "tool_analyze_summary"
                        else "tool_query_conversation"
                    )
                    state = self._nodes[tool_name](state)
                else:
                    node_fn = self._nodes[next_node]
                    state = (
                        await node_fn(state)
                        if _asyncio.iscoroutinefunction(node_fn)
                        else node_fn(state)
                    )
            return state

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
        cfg = None
        last_route_kind = ""
        last_route_model = "openai:gemma4:e4b"
        last_reply_model = ""
        last_tool_name = ""

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
            assert user_prompt == "show latest sessions"
            assert chat_history == [{"role": "user", "content": "show latest sessions"}]
            assert latest_session_id == ""
            self.last_route_kind = "tool_list_sessions"
            self.last_tool_name = ""
            return ["tool_list_sessions", "reply_fast"]

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

        async def respond_deep(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            raise AssertionError("Deep responder should not run in tool path test.")

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch(
            "pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=FakeChatAgent()
        ),
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
        "langgraph-plan",
        "langgraph-dispatch",
        "langgraph-tool-list-sessions",
        "langgraph-dispatch",
        "langgraph-respond-fast",
        "langgraph-dispatch",
    ]
    assert ("attr", "langgraph-dispatch", "output.route_kind", "tool_list_sessions") in span_events
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
            import asyncio as _asyncio

            state = dict(payload)
            state = self._nodes["human_input"](state)
            state = await self._nodes["plan"](state)
            while True:
                state = self._nodes["dispatch"](state)
                next_node = self._route_fn(state)
                if next_node == "__end__":
                    break
                if next_node == "extract_session_id":
                    state = await self._nodes["extract_session_id"](state)
                    route_kind = (state.get("session_state") or {}).get("route_kind", "")
                    tool_name = (
                        "tool_analyze_summary"
                        if route_kind == "tool_analyze_summary"
                        else "tool_query_conversation"
                    )
                    state = self._nodes[tool_name](state)
                else:
                    node_fn = self._nodes[next_node]
                    state = (
                        await node_fn(state)
                        if _asyncio.iscoroutinefunction(node_fn)
                        else node_fn(state)
                    )
            return state

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
        cfg = None
        last_route_kind = ""
        last_route_model = "openai:gemma4:e4b"
        last_reply_model = ""
        last_tool_name = ""

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
            assert latest_session_id == "oci-20260416"
            assert latest_generated_content.startswith("# Executive Report")
            self.last_route_kind = "tool_save_to_siyuan"
            return ["tool_save_to_siyuan", "reply_fast"]

        async def respond_fast(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            self.last_reply_model = "openai:gemma4:e4b"
            return f"Saved to SiYuan: {tool_output}"

        async def respond_deep(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            raise AssertionError("Deep responder should not run in save path test.")

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch(
            "pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=FakeChatAgent()
        ),
        patch("pawn_agent.core.langgraph_chat.build_phoenix_tracer", return_value=FakeTracer()),
        patch(
            "pawn_agent.core.langgraph_tools.run_save_to_siyuan_tool",
            return_value="siyuan://blocks/doc-123",
        ),
    ):
        session = asyncio.run(LangGraphChatSession.create(cfg=cfg, emit=lambda _text: None))
        session._state["durable_facts"]["latest_session_id"] = "oci-20260416"
        session._state["artifacts"]["latest_generated_content"] = "# Executive Report\n\nHello"
        session._state["artifacts"]["latest_generated_title"] = "Executive Report"
        reply = asyncio.run(session.handle_user_input("save the analysis into siyuan"))

    assert reply == "Saved to SiYuan: siyuan://blocks/doc-123"
    span_names = [event[1] for event in span_events if event[0] == "enter"]
    assert span_names == [
        "langgraph-chat-turn",
        "langgraph-human-input",
        "langgraph-plan",
        "langgraph-dispatch",
        "langgraph-tool-save-to-siyuan",
        "langgraph-dispatch",
        "langgraph-respond-fast",
        "langgraph-dispatch",
    ]
    assert ("attr", "langgraph-tool-save-to-siyuan", "tool.name", "save_to_siyuan") in span_events
    assert ("attr", "langgraph-tool-save-to-siyuan", "session.id", "oci-20260416") in span_events
    assert ("attr", "langgraph-chat-turn", "tool.name", "save_to_siyuan") in span_events


def test_langgraph_chat_session_traces_full_state_when_enabled(tmp_path: Path) -> None:
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
            import asyncio as _asyncio

            state = dict(payload)
            state = self._nodes["human_input"](state)
            state = await self._nodes["plan"](state)
            while True:
                state = self._nodes["dispatch"](state)
                next_node = self._route_fn(state)
                if next_node == "__end__":
                    break
                node_fn = self._nodes[next_node]
                state = (
                    await node_fn(state)
                    if _asyncio.iscoroutinefunction(node_fn)
                    else node_fn(state)
                )
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
            if start == "dispatch":
                self._route_fn = fn

        def compile(self):
            graph = FakeCompiledGraph(self._nodes)
            graph.set_route_fn(self._route_fn)
            return graph

    class FakeChatAgent:
        cfg = None
        last_route_kind = ""
        last_route_model = "openai:gemma4:e4b"
        last_reply_model = ""
        last_tool_name = ""

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
            self.last_route_kind = "reply_fast"
            return ["reply_fast"]

        async def respond_fast(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            self.last_reply_model = "openai:gemma4:e4b"
            return "ok"

        async def respond_deep(
            self,
            user_prompt: str,
            chat_history: list[dict[str, str]],
            *,
            tool_output: str = "",
        ) -> str:
            raise AssertionError("Deep responder should not run in this test.")

    with (
        patch(
            "pawn_agent.core.langgraph_chat._import_langgraph_core",
            return_value=(FakeStateGraph, "__start__", "__end__"),
        ),
        patch(
            "pawn_agent.core.langgraph_chat.LangGraphRouterChatAgent", return_value=FakeChatAgent()
        ),
        patch("pawn_agent.core.langgraph_chat.build_phoenix_tracer", return_value=FakeTracer()),
    ):
        session = asyncio.run(
            LangGraphChatSession.create(
                cfg=cfg,
                emit=lambda _text: None,
                trace_full_state=True,
            )
        )
        reply = asyncio.run(session.handle_user_input("hello"))

    assert reply == "ok"
    full_state_attrs = [
        event for event in span_events if event[0] == "attr" and str(event[2]).startswith("state.")
    ]
    assert full_state_attrs
    assert any('"session_state"' in str(event[3]) for event in full_state_attrs)
